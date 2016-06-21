for i in {1..25}
do
  echo "## RUN $i ###########################################"
  echo "../bin/mlp -c ../bin/g6c.nn 3 2 16 6"
  ../bin/mlp -c ../bin/g6c.nn 3 2 16 6
  echo " "
  echo "../bin/mlp -t ../bin/g6c.nn ../training/g6c_training.csv 10000 2 0 0 -e 0.19"
  ../bin/mlp -t ../bin/g6c.nn ../training/g6c_training.csv 10000 2 0 0 -e 0.19
  echo " "
  echo "../bin/mlp -r ../bin/g6c.nn ../training/g6c_test.csv"
  ../bin/mlp -r ../bin/g6c.nn ../training/g6c_test.csv
  echo "#####################################################"
  echo " "
done
