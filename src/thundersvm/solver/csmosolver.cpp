//
// Created by jiashuai on 17-10-25.
//
#include <thundersvm/solver/csmosolver.h>
#include <thundersvm/kernel/smo_kernel.h>
#include <limits.h>
//#include <hbwmalloc.h>
#include <omp.h>
//#include <thundersvm/global.h>
using namespace svm_kernel;
//#define USE_HBW
//#define USE_SIMD

//extern long memory_size;
//extern long ins_mem_size;
int hit_num = 0;
int miss_num = 0;
//number of switch
int n_switch = 0;
//int copy_num = 0;
//int new_miss_num = 0;
void
CSMOSolver::solve(const KernelMatrix &k_mat, const SyncArray<int> &y, SyncArray<float_type> &alpha, float_type &rho,
                  SyncArray<float_type> &f_val, float_type eps, float_type Cp, float_type Cn, int ws_size,
                  float_type* kernel_value_cache, bool* in_cache, int* cacheIndex, int* insId) const {
    TIMED_SCOPE(timerObj, "solve");

    int out_max_iter = 50000;
    int numT = 20;

    int divide = 50;
    int is_front = -1;
    //avoid infinite loop of repeated local diff
    int same_local_diff_cnt = 0;
    float_type previous_local_diff = INFINITY;
    int swap_local_diff_cnt = 0;
    float_type last_local_diff = INFINITY;
    float_type second_last_local_diff = INFINITY;
    long long local_iter = 0;

    int n_instances = k_mat.n_instances();
    std::cout<<"instances:"<<n_instances<<std::endl;

    int lfu_hit_num = 0;
    int lru_hit_num = 0;
    int lru_hit_num_last_time = 0;
    int lfu_hit_num_last_time = 0;
//    vector<vector<int>> reuse_dis(n_instances);
//    int reuse_dis[n_instances];
    int *reuse_dis = new int[n_instances];
    memset(reuse_dis, -1, n_instances * sizeof(int));
    int seg_size[100]; // 1-5, 6-10...
    memset(seg_size, 0, 100 * sizeof(int));
    bool use_lru = 0;




//	vector<int> hit_iter[n_instances];
//	vector<int> miss_iter[n_instances];

//    bool* shown= new bool [n_instances];
//    memset(shown, 0, sizeof(bool) * n_instances);
    bool use_hbw = 0;
    int q = ws_size / 2;
    std::cout<<"ws_size:"<<ws_size<<std::endl;
    long cache_row_size = n_instances;
    size_t cache_line_num = 5000;

    int seg_iter_size = 2 * cache_line_num / 500;
//    int n_seg_per_iter = 2 * seg_iter_size / 5;

    long hbw_size = (long)16 * 1024 * 1024 * 1024;
    //std::cout<<"size:"<<hbw_size<<std::endl;
    long ws_kernel_size = (long)ws_size * n_instances;
    std::cout<<"ws_kernel_size:"<<ws_kernel_size<<std::endl;
    long k_mat_rows_size = ws_kernel_size * sizeof(float_type);
    float_type *k_mat_rows = new float_type[ws_kernel_size];
//    float_type *kernel_record = new float_type[kernel_record_size]; //store high frequency used kernel value


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
    SyncArray<float_type> diff(2);
    float_type *diff_data = diff.host_data();

    float_type *k_mat_rows_first_half = k_mat_rows;
    float_type *k_mat_rows_last_half = k_mat_rows + ws_kernel_size / 2;
    int *used_num;


    bool *in_choose;
//    int *used_num = new int[n_instances]; //number of kernel row value being used
//    bool *in_cache = new bool[n_instances];//whether kernel row value in cache
//    int *cacheIndex = new int[n_instances];//index of kernel row value in kernel_record
//    bool *in_choose = new bool[n_instances];
    used_num = (int *) malloc(n_instances * sizeof(int));


    in_choose = (bool *) malloc(n_instances * sizeof(bool));
    int free_cache_index = 0;
    bool cache_full = false;
    memset(in_choose, 0, sizeof(bool) * n_instances);

    //int hit_num = 0;
    //int miss_num = 0;
    //int copy_num = 0;


    int *time_used = new int[cache_line_num];
    memset(time_used, -1, sizeof(int) * cache_line_num);


    memset(used_num, 0, sizeof(int) * n_instances);



    int *f_idx_data = f_idx.host_data();
    for (int i = 0; i < n_instances; ++i) {
        f_idx_data[i] = i;
    }
    init_f(alpha, y, k_mat, f_val);
    int iter = 0;
    int thre = 0;
    LOG(INFO) << "training start";
    {
        TIMED_SCOPE(timerObj, "train record time");
        int max_iter = max(100000, ws_size > INT_MAX / 100 ? INT_MAX : 100 * ws_size);
        //vector<vector <int>> ins_rec(n_instances);
        vector<int> working_set_cal_last_half;
        SyncArray<int> working_set_cal_rank(ws_size);
        int *working_set_data = working_set.host_data();
        int *working_set_cal_rank_data = working_set_cal_rank.host_data();
        //float *k_mat_rows_data = k_mat_rows.host_data();
        vector <int> recal_first_half_kernel;
//    float_type *f_idx2sort_data = f_idx2sort.host_data();
//    float_type *f_val2sort_data = f_val2sort.host_data();
        for (iter = 1;; ++iter) {
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
            if (1 == iter) {
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

                local_iter += diff_data[1];
                {
                    TIMED_SCOPE(timerObj, "update cache");
#ifdef USE_SIMD
#pragma omp simd
#endif
                    for(int i = 0; i < ws_size; i++) {
                        used_num[working_set_data[i]]++;
                        in_choose[working_set_data[i]] = 1;
//            shown[working_set_data[i]] = 1;

                        reuse_dis[working_set_data[i]]=iter;

                    }

                    miss_num+=ws_size;
                    int free_num = cache_line_num;
                    //int load_num = min(free_num, working_set_cal_last_half.size());
                    if(free_num >  ws_size){
#pragma omp parallel for
                        for(int i = 0; i < ws_size; i++){
//                        miss_num++;
//                        copy_num++;
                            memcpy(kernel_value_cache + (long)(free_cache_index + i) * cache_row_size,
                                   k_mat_rows + (long)i * n_instances,
                                   n_instances * sizeof(float_type));
                            int wsi = working_set_data[i];
                            in_cache[wsi] = true;
                            cacheIndex[wsi] = free_cache_index + i;

                            insId[free_cache_index + i] = wsi;
                            time_used[free_cache_index + i] = iter;
                        }
                        free_cache_index += ws_size;
                    }
                    else{
#pragma omp parallel for
                        for(int i = 0; i < free_num; i++){
//                        miss_num++;
//                        copy_num++;
                            memcpy(kernel_value_cache + (long)(free_cache_index + i) * cache_row_size,
                                   k_mat_rows + (long)i * n_instances,
                                   n_instances * sizeof(float_type));
                            int wsi = working_set_data[i];
                            in_cache[wsi] = true;
                            cacheIndex[wsi] = free_cache_index + i;

                            insId[free_cache_index + i] = wsi;
                            time_used[free_cache_index + i] = iter;
                        }
                        free_cache_index += free_num;
                        cache_full = true;

                        for (int i = free_num; i < ws_size; i++) {
//                            miss_num++;
                            int wsi = working_set_data[i];
                            for (int j = 0; j < n_instances; j++) {
                                if (in_cache[j] && (used_num[j] < (used_num[wsi] - thre)) && (in_choose[j] == 0)) {
                                    in_cache[j] = false;
//                                    copy_num++;
                                    memcpy(kernel_value_cache + (long)cacheIndex[j] * cache_row_size,
                                           k_mat_rows + (long) i * n_instances,
                                           n_instances * sizeof(float));
                                    in_cache[wsi] = true;
                                    cacheIndex[wsi] = cacheIndex[j];

                                    insId[cacheIndex[j]] = wsi;
                                    time_used[cacheIndex[j]] = iter;
                                    break;
                                }
                            }
                        }

                    }
                }
#ifdef USE_SIMD
#pragma omp simd
#endif
                for(int i = 0; i < ws_size; i++)
                    working_set_cal_rank_data[i] = i;

            } else {
                memset(in_choose, 0, sizeof(bool) * n_instances);
                working_set_first_half.copy_from(working_set_last_half);
#ifdef USE_SIMD
#pragma omp simd
#endif
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
#ifdef USE_SIMD
#pragma omp simd
#endif
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
                            memcpy(k_mat_rows_first_half + (long)i * n_instances, k_mat_rows_last_half + (long)i * n_instances, n_instances * sizeof(float_type));
                    }
                    //memcpy(k_mat_rows_first_half, k_mat_rows_last_half, ws_kernel_size / 2 * sizeof(float_type));

                    working_set_cal_last_half.clear();
                    rank = ws_size / 2;
                    //int first_off = 0;
                    for(int i = q; i < ws_size; i++){
                        in_choose[working_set_data[i]] = 1;
                        if(in_cache[working_set_data[i]]){
                            hit_num++;
                            lfu_hit_num++;
                            lru_hit_num++;
                            working_set_cal_rank_data[i] = -1;
//                    hit_iter[working_set_data[i]].push_back(iter);
                            time_used[cacheIndex[working_set_data[i]]] = iter;
                            //numOfIn++;
                        }
                        else {
//                    if(shown[working_set_data[i]])
//                        new_miss_num++;
                            miss_num++;
                            working_set_cal_rank_data[i] = rank++;
                            working_set_cal_last_half.emplace_back(working_set_data[i]);
                            // working_set_cal_last_half.push_back(working_set_data[i]);
//                    shown[working_set_data[i]] = 1;
                        }

                        if(reuse_dis[working_set_data[i]] != -1) {
                            seg_size[(iter - reuse_dis[working_set_data[i]] - 1) / 5]++;
                        }
                        reuse_dis[working_set_data[i]] = iter;
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
                               max_iter, cacheIndex, kernel_value_cache, working_set_cal_rank_data);
                }
                {
                    TIMED_SCOPE(timerObj, "update f");
                    //update f
                    update_f(f_val, alpha_diff, k_mat_rows, k_mat.n_instances(), kernel_value_cache, working_set_cal_rank_data,
                             cacheIndex, working_set_data);
                }
                local_iter += diff_data[1];
                //LOG(INFO)<<"f:"<<f_val;
                {
                    TIMED_SCOPE(timerObj, "update cache");
#ifdef USE_SIMD
#pragma omp simd
#endif
                    for(int i = 0; i < ws_size; i++)
                        used_num[working_set_data[i]]++;

                    int wsclh_size = working_set_cal_last_half.size();
                    if(!cache_full){
                        int free_num = cache_line_num - free_cache_index;
                        //int load_num = min(free_num, working_set_cal_last_half.size());
                        if(free_num >  wsclh_size){
#pragma omp parallel for num_threads(numT)
                            for(int i = 0; i < wsclh_size; i++){
//                        copy_num++;
                                memcpy(kernel_value_cache + (long)(free_cache_index + i) * cache_row_size,
                                       k_mat_rows + (long)(q + i) * n_instances,
                                       n_instances * sizeof(float_type));
                                int wsi = working_set_cal_last_half[i];
                                in_cache[wsi] = true;
                                cacheIndex[wsi] = free_cache_index + i;

                                insId[free_cache_index + i] = wsi;
                                time_used[free_cache_index + i] = iter;
                            }

                            free_cache_index += wsclh_size;
                        }
                        else{
                            //                std::cout<<"fulling"<<std::endl;
#pragma omp parallel for num_threads(numT)
                            for(int i = 0; i < free_num; i++){
//                        copy_num++;
                                memcpy(kernel_value_cache + (long)(free_cache_index + i) * cache_row_size,
                                       k_mat_rows + (long)(q + i) * n_instances,
                                       n_instances * sizeof(float_type));
                                int wsi = working_set_cal_last_half[i];
                                in_cache[wsi] = true;
                                cacheIndex[wsi] = free_cache_index + i;

                                insId[free_cache_index + i] = wsi;
                                time_used[free_cache_index + i] = iter;
                            }

                            free_cache_index += free_num;
                            cache_full = true;
                            if(!use_lru){
                                for (int i = free_num; i < wsclh_size; i++) {
                                    int wsi = working_set_cal_last_half[i];
                                    for (int j = 0; j < n_instances; j++) {
                                        if (in_cache[j] && (used_num[j] < (used_num[wsi] - thre)) && (in_choose[j] == 0)) {
                                            in_cache[j] = false;
//                                    copy_num++;
                                            memcpy(kernel_value_cache + (long)cacheIndex[j] * cache_row_size,
                                                   k_mat_rows + (long) (q + i) * n_instances,
                                                   n_instances * sizeof(float));
                                            in_cache[wsi] = true;
                                            cacheIndex[wsi] = cacheIndex[j];

                                            insId[cacheIndex[j]] = wsi;
                                            time_used[cacheIndex[j]] = iter;
                                            break;
                                        }
                                    }
                                }
                            }
                            else{
                                for (int i = free_num; i < wsclh_size; i++) {
                                    int wsi = working_set_cal_last_half[i];
                                    int min_idx = -1;
                                    int min = time_used[0] + 1;
                                    for(int j = 0; j < cache_line_num; j++){
                                        if((time_used[j] < min) && (in_choose[insId[j]] == 0)){
                                            min = time_used[j];
                                            min_idx = j;
                                        }
                                    }
                                    //int min_idx = get_min_idx(time_used, cache_line_num);

                                    //if(in_choose[insId[min_idx]] == 0) {
                                    if(min_idx != -1){
                                        in_cache[insId[min_idx]] = false;
//                                copy_num++;
                                        memcpy(kernel_value_cache + (long)min_idx * cache_row_size,
                                               k_mat_rows + (long)(q + i) * n_instances, n_instances * sizeof(float_type));
                                        in_cache[wsi] = true;
                                        cacheIndex[wsi] = min_idx;

                                        insId[min_idx] = wsi;
                                        time_used[min_idx] = iter;
                                    }

                                }
                            }
                        }
                        //free_cache_index += load_num;
                    }
                    else{
                        if(iter % seg_iter_size == 0) {
                            if (!use_lru) {
                                int lru_hit_count_1 = 0;
//                        std::cout << "1" << std::endl;
                                for (int i = 0; i < cache_line_num / 2500; i++)
                                    lru_hit_count_1 += seg_size[i];
//                        std::cout << "2" << std::endl;
                                float lru_hit_ratio_1 = 0.0;
//                        int total_access = 0;
//                        for(int i = 0; i < n_seg_per_iter; i++)
//                            total_access += seg_size[i];
                                if (iter == seg_iter_size)
                                    lru_hit_ratio_1 = (float) lru_hit_count_1 / (ws_size + ws_size / 2 * seg_iter_size);
                                else
                                    lru_hit_ratio_1 = (float) lru_hit_count_1 / (ws_size / 2 * seg_iter_size);
//                        lru_hit_ratio_1 = (float) lru_hit_count_1 / total_access;
//                        std::cout << "lru hit ratio:" << lru_hit_ratio_1 << std::endl;
                                if(lru_hit_ratio_1 == 1){
                                    n_switch++;
                                    use_lru = 1;
                                }
                                else {
                                    if (lru_hit_count_1 > lfu_hit_num) {
//                                std::cout << "use lru" << std::endl;
                                        lru_hit_num_last_time = lru_hit_count_1;
                                        lfu_hit_num_last_time = lfu_hit_num;
                                        n_switch++;
                                        use_lru = 1;
                                    }
                                }
                            }
                            else{
//                        if(lru_hit_num < lru_hit_num_last_time)
                                if(lru_hit_num < lfu_hit_num_last_time) {
                                    n_switch++;
                                    use_lru = 0;
                                }
                            }
                            lfu_hit_num = 0;
                            lru_hit_num = 0;
                            memset(seg_size, 0, 100 * sizeof(int));
                        }
                        if(!use_lru){
#pragma omp parallel num_threads(numT)
                            {
                                int tid = omp_get_thread_num();
                                int wstep = (wsclh_size + numT - 1) / numT;
                                int wbegin = min(tid * wstep, wsclh_size );
                                int wend = min((tid + 1) * wstep, wsclh_size );
                                int nstep = (cache_line_num + numT - 1) / numT;
                                int sbegin = min(tid * nstep, (int)cache_line_num);
                                int send = min((tid + 1) * nstep, (int)cache_line_num);
                                for (int i = wbegin; i < wend; i++) {
                                    int wsi = working_set_cal_last_half[i];
                                    for (int j = sbegin; j < send; j++) {
//                    for(int j = 0; j < n_instances; j++){
                                        if ((used_num[insId[j]] < (used_num[wsi] - thre)) && (in_choose[insId[j]] == 0)) {
                                            in_cache[insId[j]] = false;
//                                    copy_num++;
                                            memcpy(kernel_value_cache + (long)j * cache_row_size,
                                                   k_mat_rows + (long) (q + i) * n_instances,
                                                   n_instances * sizeof(float));
                                            in_cache[wsi] = true;
                                            cacheIndex[wsi] = j;

                                            insId[j] = wsi;
                                            time_used[j] = iter;
                                            break;
                                        }
                                    }
                                }

                            }
                        }
                        else{
#pragma omp parallel num_threads(numT)
                            {
                                int tid = omp_get_thread_num();
                                int wstep = (wsclh_size + numT - 1) / numT;
                                int wbegin = min(tid * wstep, wsclh_size );
                                int wend = min((tid + 1) * wstep, wsclh_size );
                                int nstep = (cache_line_num + numT - 1) / numT;
                                int sbegin = min(tid * nstep, (int)cache_line_num);
                                int send = min((tid + 1) * nstep, (int)cache_line_num);
                                if(sbegin != cache_line_num) {
                                    for (int i = wbegin; i < wend; i++) {
                                        int min_idx = -1;
                                        int min = time_used[sbegin] + 1;
                                        int wsi = working_set_cal_last_half[i];
                                        for (int j = sbegin; j < send; j++) {

                                            if ((time_used[j] < min) && (in_choose[insId[j]] == 0)) {
                                                min = time_used[j];
                                                min_idx = j;
                                            }
                                        }

                                        if (min_idx != -1) {
                                            in_cache[insId[min_idx]] = false;
//                                    copy_num++;
                                            memcpy(kernel_value_cache + (long)min_idx * cache_row_size,
                                                   k_mat_rows + (long) (q + i) * n_instances, n_instances * sizeof(float));
                                            in_cache[wsi] = true;
                                            cacheIndex[wsi] = min_idx;

                                            insId[min_idx] = wsi;
                                            time_used[min_idx] = iter;
                                        }
                                    }
                                }

                            }
                        }


                    }
                }

                //track unchanged diff
                if (fabs(diff_data[0] - previous_local_diff) < eps * 0.001) {
                    same_local_diff_cnt++;
                } else {
                    same_local_diff_cnt = 0;
                    previous_local_diff = diff_data[0];
                }

                //track unchanged swapping diff
                if(fabs(diff_data[0] - second_last_local_diff) < eps * 0.001){
                    swap_local_diff_cnt++;
                } else {
                    swap_local_diff_cnt = 0;
                }
                second_last_local_diff = last_local_diff;
                last_local_diff = diff_data[0];
                if (iter % 100 == 0)
                    LOG(INFO) << "global iter = " << iter << ", total local iter = " << local_iter << ", diff = "
                              << diff_data[0];

                if ((same_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps) || diff_data[0] < eps ||
                    (iter == out_max_iter) ||
                    (swap_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps)) {
                    rho = calculate_rho(f_val, y, alpha, Cp, Cn);
                    LOG(INFO) << "global iter = " << iter << ", total local iter = " << local_iter << ", diff = "
                              << diff_data[0];
                    LOG(INFO) << "training finished";
                    float_type obj = calculate_obj(f_val, alpha, y);
                    LOG(INFO) << "obj = " << obj;
                    break;
                }

            }
        }

        printf("\n");

        free(used_num);


        free(in_choose);
        delete[] time_used;
        //free(k_mat_rows);
        //free(kernel_record);
        delete[] k_mat_rows;
//        delete[] kernel_record;
        delete[] reuse_dis;
//    delete[] shown;

//    }
    }
//    std::cout << "iter num:" << iter << std::endl;
    std::cout<<"hit num:"<<hit_num<<std::endl;
    std::cout<<"miss num:"<<miss_num<<std::endl;
//    std::cout<<"copy num:"<<copy_num<<std::endl;
//    std::cout<<"new miss num:"<<new_miss_num<<std::endl;
    std::cout<<"hit ratio:"<<1.0 * hit_num / (hit_num + miss_num) << std::endl;
    std::cout<<"number of switches:"<<n_switch<<std::endl;
}



void
CSMOSolver::solve(const KernelMatrix &k_mat, const SyncArray<int> &y, SyncArray<float_type> &alpha, float_type &rho,
                  SyncArray<float_type> &f_val, float_type eps, float_type Cp, float_type Cn, int ws_size) const {
    TIMED_SCOPE(timerObj, "solve");

    int out_max_iter = 50000;
    int numT = 20;

    int divide = 50;
    int is_front = -1;
    //avoid infinite loop of repeated local diff
    int same_local_diff_cnt = 0;
    float_type previous_local_diff = INFINITY;
    int swap_local_diff_cnt = 0;
    float_type last_local_diff = INFINITY;
    float_type second_last_local_diff = INFINITY;
    long long local_iter = 0;

    int n_instances = k_mat.n_instances();
    std::cout<<"instances:"<<n_instances<<std::endl;

    int lfu_hit_num = 0;
    int lru_hit_num = 0;
    int lru_hit_num_last_time = 0;
    int lfu_hit_num_last_time = 0;
//    vector<vector<int>> reuse_dis(n_instances);
//    int reuse_dis[n_instances];
    int *reuse_dis = new int[n_instances];
    memset(reuse_dis, -1, n_instances * sizeof(int));
    int seg_size[100]; // 1-5, 6-10...
    memset(seg_size, 0, 100 * sizeof(int));
    bool use_lru = 0;

    bool use_hbw = 0;
    int q = ws_size / 2;
    std::cout<<"ws_size:"<<ws_size<<std::endl;
    long cache_row_size = n_instances;
    size_t cache_line_num = 5000;

    int seg_iter_size = 2 * cache_line_num / 500;
//    int n_seg_per_iter = 2 * seg_iter_size / 5;

    long hbw_size = (long)16 * 1024 * 1024 * 1024;
    //std::cout<<"size:"<<hbw_size<<std::endl;
    long ws_kernel_size = (long)ws_size * n_instances;
    std::cout<<"ws_kernel_size:"<<ws_kernel_size<<std::endl;
    long kernel_record_size = (long)cache_line_num * n_instances;
    std::cout<<"kernel_record_size:"<<kernel_record_size<<std::endl;
    long k_mat_rows_size = ws_kernel_size * sizeof(float_type);
    float_type *k_mat_rows = new float_type[ws_kernel_size];
    float_type *kernel_record = new float_type[kernel_record_size]; //store high frequency used kernel value



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
    SyncArray<float_type> diff(2);
    float_type *diff_data = diff.host_data();

    float_type *k_mat_rows_first_half = k_mat_rows;
    float_type *k_mat_rows_last_half = k_mat_rows + ws_kernel_size / 2;
    int *used_num;
    bool *in_cache;
    int *cacheIndex;
    bool *in_choose;
//    int *used_num = new int[n_instances]; //number of kernel row value being used
//    bool *in_cache = new bool[n_instances];//whether kernel row value in cache
//    int *cacheIndex = new int[n_instances];//index of kernel row value in kernel_record
//    bool *in_choose = new bool[n_instances];
    used_num = (int *) malloc(n_instances * sizeof(int));
    in_cache = (bool *) malloc(n_instances * sizeof(bool));
    cacheIndex = (int *) malloc(n_instances * sizeof(int));
    in_choose = (bool *) malloc(n_instances * sizeof(bool));
    int free_cache_index = 0;
    bool cache_full = false;
    memset(in_choose, 0, sizeof(bool) * n_instances);

    //int hit_num = 0;
    //int miss_num = 0;
    //int copy_num = 0;


    int *time_used = new int[cache_line_num];
    memset(time_used, -1, sizeof(int) * cache_line_num);
    int *insId = new int[cache_line_num];

    memset(used_num, 0, sizeof(int) * n_instances);
    memset(in_cache, 0, sizeof(bool) * n_instances);


    int *f_idx_data = f_idx.host_data();
    for (int i = 0; i < n_instances; ++i) {
        f_idx_data[i] = i;
    }
    init_f(alpha, y, k_mat, f_val);
    int iter = 0;
    int thre = 0;
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
        for (iter = 1;; ++iter) {
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
            if (1 == iter) {
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

                local_iter += diff_data[1];
                {
                    TIMED_SCOPE(timerObj, "update cache");
#ifdef USE_SIMD
#pragma omp simd
#endif
                    for(int i = 0; i < ws_size; i++) {
                        used_num[working_set_data[i]]++;
                        in_choose[working_set_data[i]] = 1;

                        reuse_dis[working_set_data[i]]=iter;

                    }

                    miss_num+=ws_size;
                    int free_num = cache_line_num;
                    if(free_num >  ws_size){
#pragma omp parallel for
                        for(int i = 0; i < ws_size; i++){
                            memcpy(kernel_record + (long)(free_cache_index + i) * cache_row_size,
                                   k_mat_rows + (long)i * n_instances,
                                   n_instances * sizeof(float_type));
                            int wsi = working_set_data[i];
                            in_cache[wsi] = true;
                            cacheIndex[wsi] = free_cache_index + i;

                            insId[free_cache_index + i] = wsi;
                            time_used[free_cache_index + i] = iter;
                        }
                        free_cache_index += ws_size;
                    }
                    else{
#pragma omp parallel for
                        for(int i = 0; i < free_num; i++){
                            memcpy(kernel_record + (long)(free_cache_index + i) * cache_row_size,
                                   k_mat_rows + (long)i * n_instances,
                                   n_instances * sizeof(float_type));
                            int wsi = working_set_data[i];
                            in_cache[wsi] = true;
                            cacheIndex[wsi] = free_cache_index + i;

                            insId[free_cache_index + i] = wsi;
                            time_used[free_cache_index + i] = iter;
                        }
                        free_cache_index += free_num;
                        cache_full = true;

//                    if((iter - divide) * is_front < 0) {
                        for (int i = free_num; i < ws_size; i++) {
//                            miss_num++;
                            int wsi = working_set_data[i];
                            for (int j = 0; j < n_instances; j++) {
                                if (in_cache[j] && (used_num[j] < (used_num[wsi] - thre)) && (in_choose[j] == 0)) {
                                    in_cache[j] = false;
//                                    copy_num++;
                                    memcpy(kernel_record + (long)cacheIndex[j] * cache_row_size,
                                           k_mat_rows + (long) i * n_instances,
                                           n_instances * sizeof(float));
                                    in_cache[wsi] = true;
                                    cacheIndex[wsi] = cacheIndex[j];

                                    insId[cacheIndex[j]] = wsi;
                                    time_used[cacheIndex[j]] = iter;
                                    break;
                                }
                            }
                        }
                    }
                }
#ifdef USE_SIMD
#pragma omp simd
#endif
                for(int i = 0; i < ws_size; i++)
                    working_set_cal_rank_data[i] = i;

            } else {
                memset(in_choose, 0, sizeof(bool) * n_instances);
                working_set_first_half.copy_from(working_set_last_half);
#ifdef USE_SIMD
#pragma omp simd
#endif
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

#ifdef USE_SIMD
#pragma omp simd
#endif
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
                    {
                        TIMED_SCOPE(timerObj, "kv copy");
#pragma omp parallel for
                        for(int i = 0; i < ws_size / 2; i++)
                            memcpy(k_mat_rows_first_half + (long)i * n_instances, k_mat_rows_last_half + (long)i * n_instances, n_instances * sizeof(float_type));
                    }

                    working_set_cal_last_half.clear();
                    rank = ws_size / 2;
                    for(int i = q; i < ws_size; i++){
                        in_choose[working_set_data[i]] = 1;
                        if(in_cache[working_set_data[i]]){
                            hit_num++;
                            lfu_hit_num++;
                            lru_hit_num++;
                            working_set_cal_rank_data[i] = -1;
                            time_used[cacheIndex[working_set_data[i]]] = iter;
                        }
                        else {
                            miss_num++;
                            working_set_cal_rank_data[i] = rank++;
                            working_set_cal_last_half.emplace_back(working_set_data[i]);

                        }

                        if(reuse_dis[working_set_data[i]] != -1) {
                            seg_size[(iter - reuse_dis[working_set_data[i]] - 1) / 5]++;
                        }
                        reuse_dis[working_set_data[i]] = iter;
                    }
                }
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
                local_iter += diff_data[1];
                //LOG(INFO)<<"f:"<<f_val;
                {
                    TIMED_SCOPE(timerObj, "update cache");
#ifdef USE_SIMD
#pragma omp simd
#endif
                    for(int i = 0; i < ws_size; i++)
                        used_num[working_set_data[i]]++;

                    int wsclh_size = working_set_cal_last_half.size();
                    if(!cache_full){
                        int free_num = cache_line_num - free_cache_index;
                        //int load_num = min(free_num, working_set_cal_last_half.size());
                        if(free_num >  wsclh_size){
#pragma omp parallel for num_threads(numT)
                            for(int i = 0; i < wsclh_size; i++){
//                        copy_num++;
                                memcpy(kernel_record + (long)(free_cache_index + i) * cache_row_size,
                                       k_mat_rows + (long)(q + i) * n_instances,
                                       n_instances * sizeof(float_type));
                                int wsi = working_set_cal_last_half[i];
                                in_cache[wsi] = true;
                                cacheIndex[wsi] = free_cache_index + i;

                                insId[free_cache_index + i] = wsi;
                                time_used[free_cache_index + i] = iter;
                            }

                            free_cache_index += wsclh_size;
                        }
                        else{
                            //                std::cout<<"fulling"<<std::endl;
#pragma omp parallel for num_threads(numT)
                            for(int i = 0; i < free_num; i++){
//                        copy_num++;
                                memcpy(kernel_record + (long)(free_cache_index + i) * cache_row_size,
                                       k_mat_rows + (long)(q + i) * n_instances,
                                       n_instances * sizeof(float_type));
                                int wsi = working_set_cal_last_half[i];
                                in_cache[wsi] = true;
                                cacheIndex[wsi] = free_cache_index + i;

                                insId[free_cache_index + i] = wsi;
                                time_used[free_cache_index + i] = iter;
                            }

                            free_cache_index += free_num;
                            cache_full = true;
                            if(!use_lru){
                                for (int i = free_num; i < wsclh_size; i++) {
                                    int wsi = working_set_cal_last_half[i];
                                    for (int j = 0; j < n_instances; j++) {
                                        if (in_cache[j] && (used_num[j] < (used_num[wsi] - thre)) && (in_choose[j] == 0)) {
                                            in_cache[j] = false;
//                                    copy_num++;
                                            memcpy(kernel_record + (long)cacheIndex[j] * cache_row_size,
                                                   k_mat_rows + (long) (q + i) * n_instances,
                                                   n_instances * sizeof(float));
                                            in_cache[wsi] = true;
                                            cacheIndex[wsi] = cacheIndex[j];

                                            insId[cacheIndex[j]] = wsi;
                                            time_used[cacheIndex[j]] = iter;
                                            break;
                                        }
                                    }
                                }
                            }
                            else{
                                for (int i = free_num; i < wsclh_size; i++) {
                                    int wsi = working_set_cal_last_half[i];
                                    int min_idx = -1;
                                    int min = time_used[0] + 1;
                                    for(int j = 0; j < cache_line_num; j++){
                                        if((time_used[j] < min) && (in_choose[insId[j]] == 0)){
                                            min = time_used[j];
                                            min_idx = j;
                                        }
                                    }
                                    //int min_idx = get_min_idx(time_used, cache_line_num);

                                    //if(in_choose[insId[min_idx]] == 0) {
                                    if(min_idx != -1){
                                        in_cache[insId[min_idx]] = false;
//                                copy_num++;
                                        memcpy(kernel_record + (long)min_idx * cache_row_size,
                                               k_mat_rows + (long)(q + i) * n_instances, n_instances * sizeof(float_type));
                                        in_cache[wsi] = true;
                                        cacheIndex[wsi] = min_idx;

                                        insId[min_idx] = wsi;
                                        time_used[min_idx] = iter;
                                    }

                                }
                            }
                        }
                        //free_cache_index += load_num;
                    }
                    else{
                        if(iter % seg_iter_size == 0) {
                            if (!use_lru) {
                                int lru_hit_count_1 = 0;
                                for (int i = 0; i < cache_line_num / 2500; i++)
                                    lru_hit_count_1 += seg_size[i];
                                float lru_hit_ratio_1 = 0.0;
                                if (iter == seg_iter_size)
                                    lru_hit_ratio_1 = (float) lru_hit_count_1 / (ws_size + ws_size / 2 * seg_iter_size);
                                else
                                    lru_hit_ratio_1 = (float) lru_hit_count_1 / (ws_size / 2 * seg_iter_size);
                                if(lru_hit_ratio_1 == 1){
                                    use_lru = 1;
                                }
                                else {
                                    if (lru_hit_count_1 > lfu_hit_num) {
//                                std::cout << "use lru" << std::endl;
                                        lru_hit_num_last_time = lru_hit_count_1;
                                        lfu_hit_num_last_time = lfu_hit_num;
                                        use_lru = 1;
                                    }
                                }
                            }
                            else{
//                        if(lru_hit_num < lru_hit_num_last_time)
                                if(lru_hit_num < lfu_hit_num_last_time)
                                    use_lru = 0;
                            }
                            lfu_hit_num = 0;
                            lru_hit_num = 0;
                            memset(seg_size, 0, 100 * sizeof(int));
                        }
                        if(!use_lru){

#pragma omp parallel num_threads(numT)
                            {
                                int tid = omp_get_thread_num();
                                int wstep = (wsclh_size + numT - 1) / numT;
                                int wbegin = min(tid * wstep, wsclh_size );
                                int wend = min((tid + 1) * wstep, wsclh_size );
                                int nstep = (cache_line_num + numT - 1) / numT;
                                int sbegin = min(tid * nstep, (int)cache_line_num);
                                int send = min((tid + 1) * nstep, (int)cache_line_num);
                                for (int i = wbegin; i < wend; i++) {
                                    int wsi = working_set_cal_last_half[i];
                                    for (int j = sbegin; j < send; j++) {
//                    for(int j = 0; j < n_instances; j++){
                                        if ((used_num[insId[j]] < (used_num[wsi] - thre)) && (in_choose[insId[j]] == 0)) {
                                            in_cache[insId[j]] = false;
//                                    copy_num++;
                                            memcpy(kernel_record + (long)j * cache_row_size,
                                                   k_mat_rows + (long) (q + i) * n_instances,
                                                   n_instances * sizeof(float));
                                            in_cache[wsi] = true;
                                            cacheIndex[wsi] = j;

                                            insId[j] = wsi;
                                            time_used[j] = iter;
                                            break;
                                        }
                                    }
                                }

                            }

                        }
                        else{
#pragma omp parallel num_threads(numT)
                            {
                                int tid = omp_get_thread_num();
                                int wstep = (wsclh_size + numT - 1) / numT;
                                int wbegin = min(tid * wstep, wsclh_size );
                                int wend = min((tid + 1) * wstep, wsclh_size );
                                int nstep = (cache_line_num + numT - 1) / numT;
                                int sbegin = min(tid * nstep, (int)cache_line_num);
                                int send = min((tid + 1) * nstep, (int)cache_line_num);
                                if(sbegin != cache_line_num) {
                                    for (int i = wbegin; i < wend; i++) {
                                        int min_idx = -1;
                                        int min = time_used[sbegin] + 1;
                                        int wsi = working_set_cal_last_half[i];
                                        for (int j = sbegin; j < send; j++) {

                                            if ((time_used[j] < min) && (in_choose[insId[j]] == 0)) {
                                                min = time_used[j];
                                                min_idx = j;
                                            }
                                        }

                                        if (min_idx != -1) {
                                            in_cache[insId[min_idx]] = false;
//                                    copy_num++;
                                            memcpy(kernel_record + (long)min_idx * cache_row_size,
                                                   k_mat_rows + (long) (q + i) * n_instances, n_instances * sizeof(float));
                                            in_cache[wsi] = true;
                                            cacheIndex[wsi] = min_idx;

                                            insId[min_idx] = wsi;
                                            time_used[min_idx] = iter;
                                        }
                                    }
                                }

                            }
                        }


                    }
                }
                //track unchanged diff
                if (fabs(diff_data[0] - previous_local_diff) < eps * 0.001) {
                    same_local_diff_cnt++;
                } else {
                    same_local_diff_cnt = 0;
                    previous_local_diff = diff_data[0];
                }

                //track unchanged swapping diff
                if(fabs(diff_data[0] - second_last_local_diff) < eps * 0.001){
                    swap_local_diff_cnt++;
                } else {
                    swap_local_diff_cnt = 0;
                }
                second_last_local_diff = last_local_diff;
                last_local_diff = diff_data[0];
                if (iter % 100 == 0)
                    LOG(INFO) << "global iter = " << iter << ", total local iter = " << local_iter << ", diff = "
                              << diff_data[0];

                if ((same_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps) || diff_data[0] < eps ||
                    (iter == out_max_iter) ||
                    (swap_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps)) {
                    rho = calculate_rho(f_val, y, alpha, Cp, Cn);
                    LOG(INFO) << "global iter = " << iter << ", total local iter = " << local_iter << ", diff = "
                              << diff_data[0];
                    LOG(INFO) << "training finished";
                    float_type obj = calculate_obj(f_val, alpha, y);
                    LOG(INFO) << "obj = " << obj;
                    break;
                }

            }
        }

        printf("\n");
        free(used_num);
        free(in_cache);
        free(cacheIndex);
        free(in_choose);
        delete[] time_used;
        delete[] insId;
        //free(k_mat_rows);
        //free(kernel_record);
        delete[] k_mat_rows;
        delete[] kernel_record;
        delete[] reuse_dis;
//    delete[] shown;

//    }
    }
//    std::cout << "iter num:" << iter << std::endl;
    std::cout<<"hit num:"<<hit_num<<std::endl;
    std::cout<<"miss num:"<<miss_num<<std::endl;
//    std::cout<<"copy num:"<<copy_num<<std::endl;
//    std::cout<<"new miss num:"<<new_miss_num<<std::endl;
    std::cout<<"hit ratio:"<<1.0 * hit_num / (hit_num + miss_num) << std::endl;

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
    double sum_free = 0;
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

float_type CSMOSolver::calculate_obj(const SyncArray<float_type> &f_val, const SyncArray<float_type> &alpha,
                                      const SyncArray<int> &y) const {
    //todo use parallel reduction for gpu and cpu
    int n_instances = f_val.size();
    float_type obj = 0;
    const float_type *f_val_data = f_val.host_data();
    const float_type *alpha_data = alpha.host_data();
    const int *y_data = y.host_data();
    for (int i = 0; i < n_instances; ++i) {
        obj += alpha_data[i] - (f_val_data[i] + y_data[i]) * alpha_data[i] * y_data[i] / 2;
    }
    return -obj;
}
