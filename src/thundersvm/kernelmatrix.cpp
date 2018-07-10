//
// Created by jiashuai on 17-9-20.
//
#include <thundersvm/svmparam.h>
#include "thundersvm/kernelmatrix.h"
#include "thundersvm/kernel/kernelmatrix_kernel.h"

using namespace svm_kernel;
KernelMatrix::KernelMatrix(const DataSet::node2d &instances, SvmParam param) {
    n_instances_ = instances.size();
    n_features_ = 0;
    this->param = param;

    //three arrays for csr representation
    vector<float_type> csr_val;
    vector<int> csr_col_ind;//index of each value of all the instances
    vector<int> csr_row_ptr(1, 0);//the start positions of the instances

    vector<float_type> csr_self_dot;
    for (int i = 0; i < n_instances_; ++i) {//convert libsvm format to csr format
        float_type self_dot = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            csr_val.push_back(instances[i][j].value);
            self_dot += instances[i][j].value * instances[i][j].value;
            csr_col_ind.push_back(instances[i][j].index - 1);//libSVM data format is one-based, convert to zero-based
            if (instances[i][j].index > n_features_) n_features_ = instances[i][j].index;
        }
        csr_row_ptr.push_back(csr_row_ptr.back() + instances[i].size());
        csr_self_dot.push_back(self_dot);
    }

    //three arrays (on GPU/CPU) for csr representation
    val_.resize(csr_val.size());
    col_ind_.resize(csr_col_ind.size());
    row_ptr_.resize(csr_row_ptr.size());
    //copy data to the three arrays
    val_.copy_from(csr_val.data(), val_.size());
    col_ind_.copy_from(csr_col_ind.data(), col_ind_.size());
    row_ptr_.copy_from(csr_row_ptr.data(), row_ptr_.size());

    self_dot_.resize(n_instances_);
    self_dot_.copy_from(csr_self_dot.data(), self_dot_.size());

    nnz_ = csr_val.size();//number of nonzero

    //pre-compute diagonal elements

    diag_.resize(n_instances_);
    switch (param.kernel_type) {
        case SvmParam::RBF:
            for (int i = 0; i < n_instances_; ++i) {
                diag_.host_data()[i] = 1;//rbf kernel
            }
            break;
        case SvmParam::LINEAR:
            diag_.copy_from(self_dot_);
            break;
        case SvmParam::POLY:
            diag_.copy_from(self_dot_);
            poly_kernel(diag_, param.gamma, param.coef0, param.degree, diag_.size());
            break;
        case SvmParam::SIGMOID:
            diag_.copy_from(self_dot_);
            sigmoid_kernel(diag_, param.gamma, param.coef0, diag_.size());
        default:
            break;
    }

}

void KernelMatrix::get_rows(const SyncArray<int> &idx,
                            SyncArray<float_type> &kernel_rows) const {//compute multiple rows of kernel matrix according to idx
    CHECK_GE(kernel_rows.size(), idx.size() * n_instances_) << "kernel_rows memory is too small";
    get_dot_product(idx, kernel_rows);
    switch (param.kernel_type) {
        case SvmParam::RBF:
            RBF_kernel(idx, self_dot_, kernel_rows, idx.size(), n_instances_, param.gamma);
            break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
            poly_kernel(kernel_rows, param.gamma, param.coef0, param.degree, kernel_rows.size());
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel(kernel_rows, param.gamma, param.coef0, kernel_rows.size());
            break;
    }
}

void KernelMatrix::get_rows(const SyncArray<int> &idx,
                            float_type *kernel_rows,
                            long k_mat_rows_size) const {//compute multiple rows of kernel matrix according to idx
    CHECK_GE(k_mat_rows_size, idx.size() * n_instances_) << "kernel_rows memory is too small";
    get_dot_product(idx, kernel_rows);
    switch (param.kernel_type) {
        case SvmParam::RBF:
            RBF_kernel(idx, self_dot_, kernel_rows, idx.size(), n_instances_, param.gamma);
            break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
            poly_kernel(kernel_rows, param.gamma, param.coef0, param.degree, k_mat_rows_size);
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel(kernel_rows, param.gamma, param.coef0, k_mat_rows_size);
            break;
    }
}

void KernelMatrix::get_rows(vector<int> &working_set_cal,
                            float_type* kernel_rows,
                            long k_mat_rows_size) const {//compute multiple rows of kernel matrix according to idx
    CHECK_GE(k_mat_rows_size, working_set_cal.size() * n_instances_) << "kernel_rows memory is too small";
    get_dot_product(working_set_cal, kernel_rows);
    switch (param.kernel_type) {
        case SvmParam::RBF:
            RBF_kernel(working_set_cal, self_dot_, kernel_rows, working_set_cal.size(), n_instances_, param.gamma);
            break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
            poly_kernel(kernel_rows, param.gamma, param.coef0, param.degree, k_mat_rows_size);
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel(kernel_rows, param.gamma, param.coef0, k_mat_rows_size);
            break;
    }
}

void KernelMatrix::get_rows(const DataSet::node2d &instances,
                            SyncArray<float_type> &kernel_rows) const {//compute the whole (sub-) kernel matrix of the given instances.
    CHECK_GE(kernel_rows.size(), instances.size() * n_instances_) << "kernel_rows memory is too small";
    get_dot_product(instances, kernel_rows);

    //compute self dot
    //TODO use thrust
    SyncArray<float_type> self_dot(instances.size());
    float_type *self_dot_data = self_dot.host_data();
    for (int i = 0; i < instances.size(); ++i) {
        float_type sum = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            sum += instances[i][j].value * instances[i][j].value;
        }
        self_dot_data[i] = sum;
    }
    switch (param.kernel_type) {
        case SvmParam::RBF:
            RBF_kernel(self_dot, self_dot_, kernel_rows, instances.size(), n_instances_, param.gamma);
            break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
            poly_kernel(kernel_rows, param.gamma, param.coef0, param.degree, kernel_rows.size());
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel(kernel_rows, param.gamma, param.coef0, kernel_rows.size());
            break;
    }
}

const SyncArray<float_type> &KernelMatrix::diag() const {
    return this->diag_;
}


void
KernelMatrix::dns_csr_mul(const SyncArray<float_type> &dense_mat, int n_rows, SyncArray<float_type> &result) const {
	TIMED_SCOPE(timerObj, "dns csr mul");
    CHECK_EQ(dense_mat.size(), n_rows * n_features_) << "dense matrix features doesn't match";
    svm_kernel::dns_csr_mul(n_instances_, n_rows, n_features_, dense_mat, val_, row_ptr_, col_ind_, nnz_, result);
}

void
KernelMatrix::dns_csr_mul(const SyncArray<float_type> &dense_mat, int n_rows, float_type *result) const {
	TIMED_SCOPE(timerObj, "dns csr mul");
    CHECK_EQ(dense_mat.size(), n_rows * n_features_) << "dense matrix features doesn't match";
    svm_kernel::dns_csr_mul(n_instances_, n_rows, n_features_, dense_mat, val_, row_ptr_, col_ind_, nnz_, result);
}

void KernelMatrix::get_dot_product(const SyncArray<int> &idx, SyncArray<float_type> &dot_product) const {
    SyncArray<float_type> data_rows(idx.size() * n_features_);
    data_rows.mem_set(0);
    get_working_set_ins(val_, col_ind_, row_ptr_, idx, data_rows, idx.size());
    dns_csr_mul(data_rows, idx.size(), dot_product);
}

void KernelMatrix::get_dot_product(const SyncArray<int> &idx, float_type* dot_product) const {
    SyncArray<float_type> data_rows(idx.size() * n_features_);
    data_rows.mem_set(0);
    get_working_set_ins(val_, col_ind_, row_ptr_, idx, data_rows, idx.size());
    dns_csr_mul(data_rows, idx.size(), dot_product);
}

void KernelMatrix::get_dot_product(vector<int> &working_set_cal, float_type* dot_product) const {
    SyncArray<float_type> data_rows(working_set_cal.size() * n_features_);
    data_rows.mem_set(0);
    get_working_set_ins(val_, col_ind_, row_ptr_, working_set_cal, data_rows, working_set_cal.size());
    dns_csr_mul(data_rows, working_set_cal.size(), dot_product);
}


void KernelMatrix::get_dot_product(const DataSet::node2d &instances, SyncArray<float_type> &dot_product) const {
    SyncArray<float_type> dense_ins(instances.size() * n_features_);
    dense_ins.mem_set(0);
    float_type *dense_ins_data = dense_ins.host_data();
    for (int i = 0; i < instances.size(); ++i) {
        float_type sum = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            if (instances[i][j].index - 1 < n_features_) {
                dense_ins_data[(instances[i][j].index - 1) * instances.size() + i] = instances[i][j].value;
                sum += instances[i][j].value * instances[i][j].value;
            } else {
//                LOG(WARNING)<<"the number of features in testing set is larger than training set";
            }
        }
    }
    dns_csr_mul(dense_ins, instances.size(), dot_product);
}


