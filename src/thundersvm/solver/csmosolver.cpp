//
// Created by jiashuai on 17-10-25.
//
#include <thundersvm/solver/csmosolver.h>
#include <thundersvm/kernel/smo_kernel.h>
#include <limits.h>
#include <hbwmalloc.h>
#include <omp.h>
using namespace svm_kernel;




void
CSMOSolver::solve(const KernelMatrix &k_mat, const SyncArray<int> &y, SyncArray<float_type> &alpha, float_type &rho,
                  SyncArray<float_type> &f_val, float_type eps, float_type Cp, float_type Cn, int ws_size) const {
	TIMED_SCOPE(timerObj, "solve");

int n_instances = k_mat.n_instances();
    int q = ws_size / 2;

    SyncArray<int> working_set(ws_size);
    SyncArray<int> working_set_first_half(q);
    SyncArray<int> working_set_last_half(q);
#ifdef USE_CUDA
    working_set_first_half.set_device_data(working_set.device_data());
    working_set_last_half.set_device_data(&working_set.device_data()[q]);
#endif
    working_set_first_half.set_host_data(working_set.host_data());
    working_set_last_half.set_host_data(&working_set.host_data()[q]);

    SyncArray<int> f_idx(n_instances);
    SyncArray<int> f_idx2sort(n_instances);
    SyncArray<float_type> f_val2sort(n_instances);
    SyncArray<float_type> alpha_diff(ws_size);
    SyncArray<float_type> diff(1);


    long cache_row_size = n_instances;
    long cache_line_num;

    long hbw_size = (long)16 * 1024 * 1024 * 1024;
    //std::cout<<"size:"<<hbw_size<<std::endl;
    long ws_kernel_size = ws_size * n_instances;
    long k_mat_rows_size = ws_kernel_size * sizeof(float_type);
    float_type *k_mat_rows;
    float_type *kernel_record; //store high frequency used kernel value
//    int m_case;
//    if(k_mat_rows_size > hbw_size/4) {
        k_mat_rows = (float_type *) malloc(k_mat_rows_size);
        cache_line_num = hbw_size / (n_instances * sizeof(float_type));
//	cache_line_num = 6000;    
//std::cout<<"cache line num"<<cache_line_num<<std::endl;
	    //cache_line_num = ws_size * 10;
//	kernel_record = (float_type *) hbw_malloc(cache_line_num * cache_row_size * sizeof(float_type));
	kernel_record = (float_type *) malloc(cache_line_num * cache_row_size * sizeof(float_type));
//    	m_case = 0;
//    }
/*
    else {
        k_mat_rows = (float_type *) hbw_malloc(k_mat_rows_size);
        cache_line_num = (hbw_size - k_mat_rows_size) / (n_instances * sizeof(float_type));
        //cache_line_num = ws_size * 10;
	kernel_record = (float_type *) hbw_malloc(cache_line_num * cache_row_size * sizeof(float_type));
	m_case = 1;
    }
*/
    float_type *k_mat_rows_first_half = k_mat_rows;
    float_type *k_mat_rows_last_half = k_mat_rows + ws_kernel_size / 2;

    int *used_num = new int[n_instances]; //number of kernel row value being used
    bool *in_cache = new bool[n_instances];//whether kernel row value in cache
    int *cacheIndex = new int[n_instances];//index of kernel row value in kernel_record
    int free_cache_index = 0;
    bool cache_full = false;
    bool *in_choose = new bool[n_instances];
    memset(in_choose, 0, sizeof(bool) * n_instances);

	//int hit_num = 0;
	//int miss_num = 0;
	//int copy_num = 0;
    memset(used_num, 0, sizeof(int) * n_instances);
    memset(in_cache, 0, sizeof(bool) * n_instances);


    int *f_idx_data = f_idx.host_data();
    for (int i = 0; i < n_instances; ++i) {
        f_idx_data[i] = i;
    }
    init_f(alpha, y, k_mat, f_val);
    LOG(INFO) << "training start";
{
	TIMED_SCOPE(timerObj, "train record time");
    int max_iter = max(100000, ws_size > INT_MAX / 100 ? INT_MAX : 100 * ws_size);
    //vector<vector <int>> ins_rec(n_instances);
    vector <int> working_set_cal_last_half;
    SyncArray<int> working_set_cal_rank(ws_size);
    int *working_set_data = working_set.host_data();
    int *working_set_cal_rank_data = working_set_cal_rank.host_data();
    //float *k_mat_rows_data = k_mat_rows.host_data();
    vector <int> recal_first_half_kernel;
//    float_type *f_idx2sort_data = f_idx2sort.host_data();
//    float_type *f_val2sort_data = f_val2sort.host_data();
    for (int iter = 0;; ++iter) {
        //select working set
{
	TIMED_SCOPE(timerObj, "f copy");
        f_idx2sort.copy_from(f_idx);
        f_val2sort.copy_from(f_val);
}
{
TIMED_SCOPE(timerObj, "f sort");
        sort_f(f_val2sort, f_idx2sort);
}
        vector<int> ws_indicator(n_instances, 0);
        if (0 == iter) {
	{
		TIMED_SCOPE(timerObj, "select working set");
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set);
	}
	{
		TIMED_SCOPE(timerObj, "get rows");
            k_mat.get_rows(working_set, k_mat_rows, ws_kernel_size);
	}
	{
		TIMED_SCOPE(timerObj, "smo kernel");
            smo_kernel(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat.diag(), n_instances, eps,
                       diff, max_iter);
	}
	{
		TIMED_SCOPE(timerObj, "update f");
            //update f
            update_f(f_val, alpha_diff, k_mat_rows, k_mat.n_instances());
	}
	{
		TIMED_SCOPE(timerObj, "update cache");
#pragma omp simd
        for(int i = 0; i < ws_size; i++) {
            used_num[working_set_data[i]]++;
            in_choose[working_set_data[i]] = 1;
        }

//            for(int i = 0; i < ws_size; i++){
//                int wsi = working_set_data[i];
//
//                if(free_cache_index == cache_line_num)
//                    cache_full = true;
//                if(cache_full){
//                    for(int j = 0; j < n_instances; j++){
//                        if((used_num[j] < used_num[wsi]) && in_cache[j]){
//                            in_cache[j] = false;
//                            memcpy(kernel_record + cacheIndex[j] * cache_row_size,
//                                   k_mat_rows + i * n_instances, n_instances * sizeof(float));
//                            in_cache[wsi] = true;
//                            cacheIndex[wsi] = cacheIndex[j];
//                            break;
//                        }
//                    }
//                }
//                else{
//                    memcpy(kernel_record + free_cache_index * cache_row_size,
//                           k_mat_rows + i * n_instances, n_instances * sizeof(float));
//                    in_cache[wsi] = true;
//                    cacheIndex[wsi] = free_cache_index;
//                    free_cache_index++;
//                }
//            }
                int free_num = cache_line_num;
                //int load_num = min(free_num, working_set_cal_last_half.size());
                if(free_num >  ws_size){
#pragma omp parallel
                    for(int i = 0; i < ws_size; i++){
                        //miss_num++;
			//copy_num++;
			memcpy(kernel_record + (free_cache_index + i) * cache_row_size,
                               k_mat_rows + i * n_instances,
                               n_instances * sizeof(float_type));
                        int wsi = working_set_data[i];
                        in_cache[wsi] = true;
                        cacheIndex[wsi] = free_cache_index + i;
                    }
                    free_cache_index += ws_size;
                }
                else{
#pragma omp parallel
                    for(int i = 0; i < free_num; i++){
                        //miss_num++;
			//copy_num++;
			memcpy(kernel_record + (free_cache_index + i) * cache_row_size,
                               k_mat_rows + i * n_instances,
                               n_instances * sizeof(float_type));
                        int wsi = working_set_data[i];
                        in_cache[wsi] = true;
                        cacheIndex[wsi] = free_cache_index + i;
                    }
                    free_cache_index += free_num;
                    cache_full = true;
                    for(int i = free_num; i < ws_size; i++){
                        //miss_num++;
			int wsi = working_set_data[i];
                        for(int j = 0; j < n_instances; j++){
                            if(in_cache[j] && (used_num[j] < used_num[wsi]) && (in_choose[j] == 0)){
                                in_cache[j] = false;
                                //copy_num++;
				memcpy(kernel_record + cacheIndex[j] * cache_row_size,
                                       k_mat_rows + i * n_instances,
                                       n_instances * sizeof(float));
                                in_cache[wsi] = true;
                                cacheIndex[wsi] = cacheIndex[j];
                                break;
                            }
                        }
                    }
                }
	}
#pragma omp simd
            for(int i = 0; i < ws_size; i++)
                working_set_cal_rank_data[i] = i;

        } else {
            memset(in_choose, 0, sizeof(bool) * n_instances);
            working_set_first_half.copy_from(working_set_last_half);
#pragma omp simd
            for (int i = 0; i < q; ++i) {
                ws_indicator[working_set_data[i]] = 1;
            }

	{
		TIMED_SCOPE(timerObj, "select working set");
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set_last_half);
	}
{
	TIMED_SCOPE(timerObj, "ws&kv setup");
            int rank = 0;
            //int reuse_num_first_half = 0;

            //int numOfIn = q;
#pragma omp simd
            for(int i = 0; i < q; i++){
                int last_half = working_set_cal_rank_data[i + q];
                if(last_half != -1) {
			working_set_cal_rank_data[i] = last_half - q;
                    //reuse_num_first_half++;
                }
                else{
                    working_set_cal_rank_data[i] = -1;
	    	}
            }
            //k_mat_rows_first_half.copy_from(k_mat_rows_last_half);
{
	TIMED_SCOPE(timerObj, "kv copy");
#pragma omp parallel for
            for(int i = 0; i < ws_size / 2; i++)
		memcpy(k_mat_rows_first_half + i * n_instances, k_mat_rows_last_half + i * n_instances, n_instances * sizeof(float_type));   
}
	//memcpy(k_mat_rows_first_half, k_mat_rows_last_half, ws_kernel_size / 2 * sizeof(float_type));

            working_set_cal_last_half.clear();
            rank = ws_size / 2;
            //int first_off = 0;
            for(int i = q; i < ws_size; i++){
                    in_choose[working_set_data[i]] = 1;
                if(in_cache[working_set_data[i]]){
		//	hit_num++;
                    working_set_cal_rank_data[i] = -1;
                    //numOfIn++;
                }
                else {
		//	miss_num++;
                   // if(!first_off){
                   //     first_off = i;
                   // }
                    working_set_cal_rank_data[i] = rank++;
                    working_set_cal_last_half.emplace_back(working_set_data[i]);
                   // working_set_cal_last_half.push_back(working_set_data[i]);
                }
            }
}
    //        std::cout<<"iter"<<iter<<":"<<(float_type)numOfIn/ws_size<<std::endl;
    //        std::cout<<"size:"<<working_set_cal_last_half.size()<<std::endl;
            if(working_set_cal_last_half.size())
	{
		TIMED_SCOPE(timerObj, "get rows");
            k_mat.get_rows(working_set_cal_last_half, k_mat_rows_last_half, ws_kernel_size / 2);
	}
        {
		TIMED_SCOPE(timerObj, "smo kernel");
	    //local smo
            smo_kernel(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat.diag(), n_instances, eps, diff,
                       max_iter, cacheIndex, kernel_record, working_set_cal_rank_data);
        }
	{
		TIMED_SCOPE(timerObj, "update f");
		//update f
            update_f(f_val, alpha_diff, k_mat_rows, k_mat.n_instances(), kernel_record, working_set_cal_rank_data,
                     cacheIndex, working_set_data);
    }
            //LOG(INFO)<<"f:"<<f_val;
	{
		TIMED_SCOPE(timerObj, "update cache");
#pragma omp simd
	        for(int i = 0; i < ws_size; i++)
                used_num[working_set_data[i]]++;

//            for(int i = q; i < ws_size; i++){
//                int wsi = working_set_data[i];
//                //used_num[wsi]++;
//                if(working_set_cal_rank_data[i] != -1){
//                    if(free_cache_index == cache_line_num)
//                        cache_full = true;
//                    if(cache_full){
//                        for(int j = 0; j < n_instances; j++){
//                            if(in_cache[j] && (used_num[j] < used_num[wsi])){
//                                in_cache[j] = false;
//                                memcpy(kernel_record + cacheIndex[j] * cache_row_size,
//                                       k_mat_rows + working_set_cal_rank_data[i] * n_instances,
//                                       n_instances * sizeof(float));
//                                in_cache[wsi] = true;
//                                cacheIndex[wsi] = cacheIndex[j];
//                                break;
//                            }
//                        }
//                    }
//                    else{
//                        memcpy(kernel_record + free_cache_index * cache_row_size,
//                               k_mat_rows + working_set_cal_rank_data[i] * n_instances,
//                               n_instances * sizeof(float));
//                        in_cache[wsi] = true;
//                        cacheIndex[wsi] = free_cache_index;
//                        free_cache_index++;
//                    }
//                }
//            }


        int wsclh_size = working_set_cal_last_half.size();
            if(!cache_full){
                int free_num = cache_line_num - free_cache_index;
                //int load_num = min(free_num, working_set_cal_last_half.size());
                if(free_num >  wsclh_size){
#pragma omp parallel for
                    for(int i = 0; i < wsclh_size; i++){
                        //copy_num++;
			memcpy(kernel_record + (free_cache_index + i) * cache_row_size,
                               k_mat_rows + (q + i) * n_instances,
                               n_instances * sizeof(float_type));
                        int wsi = working_set_cal_last_half[i];
                        in_cache[wsi] = true;
                        cacheIndex[wsi] = free_cache_index + i;
                    }

                    //memcpy(kernel_record + free_cache_index * cache_row_size,
                    //       k_mat_rows + q * n_instances,
                    //       working_set_cal_last_half.size() * n_instances * sizeof(float_type));

                    //int free_off = 0;

//#pragma omp simd
                    //for(int i = 0; i < working_set_cal_last_half.size(); i++){
                    //    int wsi = working_set_cal_last_half[i];
                    //    in_cache[wsi] = true;
                    //    cacheIndex[wsi] = free_cache_index + i;
                        //free_cache_index++;
                    //}

                    free_cache_index += wsclh_size;
                }
                else{
                    //std::cout<<"fulling"<<std::endl;
#pragma omp parallel for
                    for(int i = 0; i < free_num; i++){
                        //copy_num++;
			memcpy(kernel_record + (free_cache_index + i) * cache_row_size,
                               k_mat_rows + (q + i) * n_instances,
                               n_instances * sizeof(float_type));
                        int wsi = working_set_cal_last_half[i];
                        in_cache[wsi] = true;
                        cacheIndex[wsi] = free_cache_index + i;
                    }

                    //memcpy(kernel_record + free_cache_index * cache_row_size,
                    //       k_mat_rows + q * n_instances,
                    //       free_num * n_instances * sizeof(float_type));


//#pragma omp simd
                    //for(int i = 0; i < free_num; i++){
                    //    int wsi = working_set_cal_last_half[i];
                    //   in_cache[wsi] = true;
                    //    cacheIndex[wsi] = free_cache_index + i;
                        //free_cache_index++;
                    //}

                    free_cache_index += free_num;
                    cache_full = true;
                    for(int i = free_num; i < wsclh_size; i++){
                        int wsi = working_set_cal_last_half[i];
                        for(int j = 0; j < n_instances; j++){
                            if(in_cache[j] && (used_num[j] < used_num[wsi]) && (in_choose[j] == 0)){
                                in_cache[j] = false;
                                //copy_num++;
				memcpy(kernel_record + cacheIndex[j] * cache_row_size,
                                       k_mat_rows + (q + i) * n_instances,
                                       n_instances * sizeof(float));
                                in_cache[wsi] = true;
                                cacheIndex[wsi] = cacheIndex[j];
                                break;
                            }
                        }
                    }
                }
                //free_cache_index += load_num;
            }
            else{
    //            std::cout<<"fulled"<<std::endl;
#pragma omp parallel for schedule(guided)
                for(int i = 0; i < wsclh_size; i++){
                    int wsi = working_set_cal_last_half[i];
                    //int tid = omp_get_thread_num();
                    //int nthread = omp_get_num_threads();
                    int nstep = (n_instances + wsclh_size - 1) / wsclh_size;
                    int sbegin = min(i * nstep, n_instances);
                    int send = min((i + 1) * nstep, n_instances);
                    for(int j = sbegin; j < send; j++){
//                    for(int j = 0; j < n_instances; j++){
                        if(in_cache[j] && (used_num[j] < used_num[wsi]) && (in_choose[j] == 0)){
                            in_cache[j] = false;
                            //copy_num++;
				            memcpy(kernel_record + cacheIndex[j] * cache_row_size,
                                   k_mat_rows + (q + i) * n_instances,
                                   n_instances * sizeof(float));
                            in_cache[wsi] = true;
                            cacheIndex[wsi] = cacheIndex[j];
                            break;
                        }
                    }
                }
            }


	}
        }
        if (iter % 10 == 0) {
            printf(".");
            std::cout.flush();
        }
        if (diff.host_data()[0] < eps) {
            rho = calculate_rho(f_val, y, alpha, Cp, Cn);
        //    std::cout<<"iter num:"<<iter<<std::endl;
            break;
        }
    }
    
    printf("\n");
/*
    if(m_case){
	hbw_free(k_mat_rows);
	hbw_free(kernel_record);
    }
*/
//    else{
	delete[] used_num;
	delete[] in_cache;
	delete[] cacheIndex;
	free(k_mat_rows);
	hbw_free(kernel_record);
//	free(kernel_record);
//    }
    }
	//std::cout<<"hit num:"<<hit_num<<std::endl;
	//std::cout<<"miss num:"<<miss_num<<std::endl;
	//std::cout<<"copy num:"<<copy_num<<std::endl;
}

void
CSMOSolver::select_working_set(vector<int> &ws_indicator, const SyncArray<int> &f_idx2sort, const SyncArray<int> &y,
                               const SyncArray<float_type> &alpha, float_type Cp, float_type Cn,
                               SyncArray<int> &working_set) const {
    int n_instances = ws_indicator.size();
    int p_left = 0;
    int p_right = n_instances - 1;
    int n_selected = 0;
    const int *index = f_idx2sort.host_data();
    const int *y_data = y.host_data();
    const float_type *alpha_data = alpha.host_data();
    int *working_set_data = working_set.host_data();
    while (n_selected < working_set.size()) {
        int i;
        if (p_left < n_instances) {
            i = index[p_left];
            while (ws_indicator[i] == 1 || !is_I_up(alpha_data[i], y_data[i], Cp, Cn)) {
                //construct working set of I_up
                p_left++;
                if (p_left == n_instances) break;
                i = index[p_left];
            }
            if (p_left < n_instances) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
        if (p_right >= 0) {
            i = index[p_right];
            while (ws_indicator[i] == 1 || !is_I_low(alpha_data[i], y_data[i], Cp, Cn)) {
                //construct working set of I_low
                p_right--;
                if (p_right == -1) break;
                i = index[p_right];
            }
            if (p_right >= 0) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }

    }
}

float_type
CSMOSolver::calculate_rho(const SyncArray<float_type> &f_val, const SyncArray<int> &y, SyncArray<float_type> &alpha,
                          float_type Cp,
                          float_type Cn) const {
    int n_free = 0;
    float_type sum_free = 0;
    float_type up_value = INFINITY;
    float_type low_value = -INFINITY;
    const float_type *f_val_data = f_val.host_data();
    const int *y_data = y.host_data();
    float_type *alpha_data = alpha.host_data();
    for (int i = 0; i < alpha.size(); ++i) {
        if (is_free(alpha_data[i], y_data[i], Cp, Cn)) {
            n_free++;
            sum_free += f_val_data[i];
        }
        if (is_I_up(alpha_data[i], y_data[i], Cp, Cn)) up_value = min(up_value, f_val_data[i]);
        if (is_I_low(alpha_data[i], y_data[i], Cp, Cn)) low_value = max(low_value, f_val_data[i]);
    }
    return 0 != n_free ? (sum_free / n_free) : (-(up_value + low_value) / 2);
}

void CSMOSolver::init_f(const SyncArray<float_type> &alpha, const SyncArray<int> &y, const KernelMatrix &k_mat,
                        SyncArray<float_type> &f_val) const {
    //todo auto set batch size
    int batch_size = 100;
    vector<int> idx_vec;
    vector<float_type> alpha_diff_vec;
    const int *y_data = y.host_data();
    const float_type *alpha_data = alpha.host_data();
    for (int i = 0; i < alpha.size(); ++i) {
        if (alpha_data[i] != 0) {
            idx_vec.push_back(i);
            alpha_diff_vec.push_back(-alpha_data[i] * y_data[i]);
        }
        if (idx_vec.size() > batch_size || (i == alpha.size() - 1 && idx_vec.size() > 0)) {
            SyncArray<int> idx(idx_vec.size());
            SyncArray<float_type> alpha_diff(idx_vec.size());
            idx.copy_from(idx_vec.data(), idx_vec.size());
            alpha_diff.copy_from(alpha_diff_vec.data(), idx_vec.size());
            SyncArray<float_type> kernel_rows(idx.size() * k_mat.n_instances());
            k_mat.get_rows(idx, kernel_rows);
            update_f(f_val, alpha_diff, kernel_rows, k_mat.n_instances());
            idx_vec.clear();
            alpha_diff_vec.clear();
        }
    }
}

void
CSMOSolver::smo_kernel(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                       SyncArray<float_type> &alpha_diff,
                       const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                       const SyncArray<float_type> &k_mat_rows,
                       const SyncArray<float_type> &k_mat_diag, int row_len, float_type eps,
                       SyncArray<float_type> &diff,
                       int max_iter) const {
    c_smo_solve(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat_diag, row_len, eps, diff, max_iter);
}

void
CSMOSolver::smo_kernel(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                       SyncArray<float_type> &alpha_diff,
                       const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                       float_type* k_mat_rows,
                       const SyncArray<float_type> &k_mat_diag, int row_len, float_type eps,
                       SyncArray<float_type> &diff,
                       int max_iter) const {
    c_smo_solve(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat_diag, row_len, eps, diff, max_iter);
}

void
CSMOSolver::smo_kernel(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                       SyncArray<float_type> &alpha_diff,
                       const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                       float_type* k_mat_rows,
                       const SyncArray<float_type> &k_mat_diag, int row_len, float_type eps,
                       SyncArray<float_type> &diff,
                       int max_iter,
                       int *cacheIndex,
                       float *kernel_record,
                       int* working_set_cal_rank_data) const {
    c_smo_solve(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat_diag, row_len, eps, diff, max_iter,
                cacheIndex, kernel_record, working_set_cal_rank_data);
}
