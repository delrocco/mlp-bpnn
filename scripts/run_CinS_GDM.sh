for i in {1..25}
do
  echo "## RUN $i ###########################################"
  echo "../bin/mlp -c ../bin/cins.nn 3 2 16 1"
  ../bin/mlp -c ../bin/cins.nn 3 2 16 1
  echo " "
  echo "../bin/mlp -t ../bin/cins.nn ../training/CinS_training.csv 5000 1 0.06 0.5"
  ../bin/mlp -t ../bin/cins.nn ../training/CinS_training.csv 5000 1 0.06 0.5
  echo " "
  echo "../bin/mlp -r ../bin/cins.nn ../training/CinS_test.csv"
  ../bin/mlp -r ../bin/cins.nn ../training/CinS_test.csv
  echo "#####################################################"
  echo " "
done
