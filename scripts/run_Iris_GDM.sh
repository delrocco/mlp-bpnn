for i in {1..25}
do
  echo "## RUN $i ###########################################"
  echo "../bin/mlp -c ../bin/iris.nn 3 2 4 1"
  ../bin/mlp -c ../bin/iris.nn 3 2 4 1
  echo " "
  echo "../bin/mlp -t ../bin/iris.nn ../training/Iris_training.csv 5000 1 0.06 0.85"
  ../bin/mlp -t ../bin/iris.nn ../training/Iris_training.csv 5000 1 0.06 0.85
  echo " "
  echo "../bin/mlp -r ../bin/iris.nn ../training/Iris_test.csv"
  ../bin/mlp -r ../bin/iris.nn ../training/Iris_test.csv
  echo "#####################################################"
  echo " "
done
