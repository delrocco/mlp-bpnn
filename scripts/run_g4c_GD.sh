for i in {1..25}
do
  echo "## RUN $i ###########################################"
  echo "../bin/mlp -c ../bin/g4c.nn 4 2 9 4"
  ../bin/mlp -c ../bin/g4c.nn 4 2 9 4
  echo " "
  echo "../bin/mlp -t ../bin/g4c.nn ../training/g4c_training.csv 2000 0 0.85 0 -e 0.22"
  ../bin/mlp -t ../bin/g4c.nn ../training/g4c_training.csv 2000 0 0.85 0 -e 0.22
  echo " "
  echo "../bin/mlp -r ../bin/g4c.nn ../training/g4c_test.csv"
  ../bin/mlp -r ../bin/g4c.nn ../training/g4c_test.csv
  echo "#####################################################"
  echo " "
done
