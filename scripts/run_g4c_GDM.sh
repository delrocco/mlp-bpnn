for i in {1..25}
do
  echo "## RUN $i ###########################################"
  echo "../bin/mlp -c ../bin/g4c.nn 4 2 9 4"
  ../bin/mlp -c ../bin/g4c.nn 4 2 9 4
  echo " "
  echo "../bin/mlp -t ../bin/g4c.nn ../training/g4c_training.csv 3000 1 0.04 0.75 -e 0.26"
  ../bin/mlp -t ../bin/g4c.nn ../training/g4c_training.csv 3000 1 0.04 0.75 -e 0.26
  echo " "
  echo "../bin/mlp -r ../bin/g4c.nn ../training/g4c_test.csv"
  ../bin/mlp -r ../bin/g4c.nn ../training/g4c_test.csv
  echo "#####################################################"
  echo " "
done
