for i in {1..25}
do
  echo "## RUN $i ###########################################"
  echo "../bin/mlp -c ../bin/iris.nn 3 2 4 1"
  ../bin/mlp -c ../bin/iris.nn 3 2 4 1
  echo " "
  echo "../bin/mlp -t ../bin/iris.nn ../training/Iris_training.csv 4000 0 0.75 0"
  ../bin/mlp -t ../bin/iris.nn ../training/Iris_training.csv 4000 0 0.75 0
  echo " "
  echo "../bin/mlp -r ../bin/iris.nn ../training/Iris_test.csv"
  ../bin/mlp -r ../bin/iris.nn ../training/Iris_test.csv
  echo "#####################################################"
  echo " "
done
