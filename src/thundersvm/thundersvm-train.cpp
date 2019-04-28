//
// Created by jiashuai on 17-9-14.
//

#include <thundersvm/util/log.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/model/svr.h>
#include <thundersvm/model/oneclass_svc.h>
#include <thundersvm/model/nusvc.h>
#include <thundersvm/model/nusvr.h>
#include <thundersvm/util/metric.h>
#include "thundersvm/cmdparser.h"
#include <unistd.h>
int main(int argc, char **argv) {
    try {
		el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);
        //el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
		CMDParser parser;
        parser.parse_command_line(argc, argv);
        DataSet train_dataset;
        bool train_multi_label = 1;
        int max_label;
        if(train_multi_label) {
            train_dataset.load_from_multi_label_file(parser.svmtrain_input_file_name);
            max_label = train_dataset.max_multi_label();
            std::cout<<"max num label:"<<max_label<<std::endl;
        }
        else
            train_dataset.load_from_file(parser.svmtrain_input_file_name);
//        std::shared_ptr<SvmModel> model;
//        switch (parser.param_cmd.svm_type) {
//            case SvmParam::C_SVC:
//                model.reset(new SVC());
//                break;
//            case SvmParam::NU_SVC:
//                model.reset(new NuSVC());
//                break;
//            case SvmParam::ONE_CLASS:
//                model.reset(new OneClassSVC());
//                break;
//            case SvmParam::EPSILON_SVR:
//                model.reset(new SVR());
//                break;
//            case SvmParam::NU_SVR:
//                model.reset(new NuSVR());
//                break;
//        }

        //todo add this to check_parameter method
        if (parser.param_cmd.svm_type == SvmParam::NU_SVC) {
            train_dataset.group_classes();
            for (int i = 0; i < train_dataset.n_classes(); ++i) {
                int n1 = train_dataset.count()[i];
                for (int j = i + 1; j < train_dataset.n_classes(); ++j) {
                    int n2 = train_dataset.count()[j];
                    if (parser.param_cmd.nu * (n1 + n2) / 2 > min(n1, n2)) {
                        printf("specified nu is infeasible\n");
                        return 1;
                    }
                }
            }
        }

        if (parser.param_cmd.gamma == 0 && parser.param_cmd.kernel_type != SvmParam::LINEAR){
            parser.param_cmd.gamma = 1.f / train_dataset.n_features();
            LOG(WARNING)<<"using default gamma="<<parser.param_cmd.gamma;
        }



#ifdef USE_CUDA
        CUDA_CHECK(cudaSetDevice(parser.gpu_id));
#endif


        //perform svm testing
//        std::shared_ptr<Metric> metric;
//        switch (parser.param_cmd.svm_type) {
//            case SvmParam::C_SVC:
//            case SvmParam::NU_SVC: {
//                metric.reset(new Accuracy());
//                break;
//            }
//            case SvmParam::EPSILON_SVR:
//            case SvmParam::NU_SVR: {
//                metric.reset(new MSE());
//                break;
//            }
//            case SvmParam::ONE_CLASS: {
//            }
//        }


        if (parser.do_cross_validation) {
            std::shared_ptr<SvmModel> model;
            switch (parser.param_cmd.svm_type) {
                case SvmParam::C_SVC:
                    model.reset(new SVC());
                    break;
                case SvmParam::NU_SVC:
                    model.reset(new NuSVC());
                    break;
                case SvmParam::ONE_CLASS:
                    model.reset(new OneClassSVC());
                    break;
                case SvmParam::EPSILON_SVR:
                    model.reset(new SVR());
                    break;
                case SvmParam::NU_SVR:
                    model.reset(new NuSVR());
                    break;
            }
//            predict_y = model->cross_validation(train_dataset, parser.param_cmd, parser.nr_fold);
        } else {
            size_t n_cache_line = 5000;
            int n_instances = train_dataset.n_instances();

//            float_type* kernel_value_cache = new float_type[cache_size];
            float overall_accuracy = 0;
            std::shared_ptr<SvmModel> model;
//            for(int i = 1; i <= max_label; i++) {
//                vector<float_type> predict_y;
//                if(!train_dataset.init_y_id(i))
//                    continue;
            switch (parser.param_cmd.svm_type) {
                case SvmParam::C_SVC:
                    model.reset(new SVC());
                    break;
                case SvmParam::NU_SVC:
                    model.reset(new NuSVC());
                    break;
                case SvmParam::ONE_CLASS:
                    model.reset(new OneClassSVC());
                    break;
                case SvmParam::EPSILON_SVR:
                    model.reset(new SVR());
                    break;
                case SvmParam::NU_SVR:
                    model.reset(new NuSVR());
                    break;
            }


            std::shared_ptr<Metric> metric;
            switch (parser.param_cmd.svm_type) {
                case SvmParam::C_SVC:
                case SvmParam::NU_SVC: {
                    metric.reset(new Accuracy());
                    break;
                }
                case SvmParam::EPSILON_SVR:
                case SvmParam::NU_SVR: {
                    metric.reset(new MSE());
                    break;
                }
                case SvmParam::ONE_CLASS: {
                }
            }


            model->is_train_multi = 1;
            if(model->is_train_multi) {
                model->init_cache(n_instances, n_cache_line);

                //only try 100 labels
                max_label = 10;
                for (int i = 1; i <= max_label; i++) {
                    vector <float_type> predict_y;
                    if (!train_dataset.init_y_id(i))
                        continue;
//                model->train_with_cache(train_dataset, parser.param_cmd, kernel_value_cache);
                    model->train(train_dataset, parser.param_cmd);
                    LOG(INFO) << "training finished";

                    continue;
                    LOG(INFO) << "evaluating training score";
                    predict_y = model->predict(train_dataset.instances(), 1000);

                    float current_accuracy;
                    if (metric) {
//                    LOG(INFO) << metric->name() << " = " << metric->score(predict_y, train_dataset.y());
                        current_accuracy = metric->score(predict_y, train_dataset.y());
                        overall_accuracy += current_accuracy;
//                        overall_accuracy += metric->score(predict_y, train_dataset.y());
                    }
                    LOG(INFO) <<"accuracy of current solver:"<<current_accuracy;
                    LOG(INFO) <<"accumulated accuracy:"<<overall_accuracy/i;

//                model.reset();
//                metric.reset();

                }
                overall_accuracy /= max_label;
//            std::cout<<"accuracy:"<<overall_accuracy<<std::endl;
                LOG(INFO) << "Accuracy = " << overall_accuracy;
            }
            else{
                model->train(train_dataset, parser.param_cmd);
                LOG(INFO)<<"training finished";
                	return 0;
                model->save_to_file(parser.model_file_name);
                LOG(INFO)<<"evaluating training score";
                vector<float_type> predict_y;
                predict_y = model->predict(train_dataset.instances(), 100);
                if (metric) {
                    LOG(INFO) << metric->name() << " = " << metric->score(predict_y, train_dataset.y());
                }
            }



        }



    }
    catch (std::bad_alloc &) {
        LOG(FATAL) << "out of host memory";
        exit(EXIT_FAILURE);
    }
    catch (std::exception const &x) {
        LOG(FATAL) << x.what();
        exit(EXIT_FAILURE);
    }
    catch (...) {
        LOG(FATAL) << "unknown error";
        exit(EXIT_FAILURE);
    }
}

