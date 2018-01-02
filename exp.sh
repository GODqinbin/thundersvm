#numactl --membind=4,5,6,7 build/bin/thundersvm-train -c 100 -g 0.5 /home/h/haofu/qinbin/thundersvm/dataset/a9a > log/a9a.log
numactl --membind=4,5,6,7 build/bin/thundersvm-train -c 10 -g 0.125 /home/h/haofu/qinbin/thundersvm/dataset/mnist.scale > log/mnist.log
#build/bin/thundersvm-train -c 100 -g 0.125 /home/qinbin/thundersvm_zeyi/thundersvm/dataset/rcv1_train.binary > log/rcv1.log
#build/bin/thundersvm-train -c 4 -g 0.5 /home/qinbin/thundersvm_zeyi/thundersvm/dataset/real-sim > log/real-sim.log
#build/bin/thundersvm-train -c 10 -g 0.5 /home/qinbin/thundersvm_zeyi/thundersvm/dataset/w8a > log/w8a.log
#build/bin/thundersvm-train -c 10 -g 0.002 /home/qinbin/thundersvm_zeyi/thundersvm/dataset/cifar10.libsvm > log/cifar10.log
#build/bin/thundersvm-train -c 1 -g 0.3 /home/qinbin/thundersvm_zeyi/thundersvm/dataset/connect-4 > log/connect-4.log
#build/bin/thundersvm-train -c 10 -g 0.125 /home/qinbin/thundersvm_zeyi/thundersvm/dataset/mnist.scale > log/mnist.log
#build/bin/thundersvm-train -c 4 -g 0.5 /home/qinbin/thundersvm_zeyi/thundersvm/dataset/news20.scale > log/news20.scale.log
