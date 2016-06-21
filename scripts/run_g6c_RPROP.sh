for i in {1..25}
do
  echo "#START###############################"
  echo "mlp -c z 3 2 16 6"
  mlp -c z 3 2 16 6
  echo " "
  echo "mlp -t z g6c_training.csv 10000 2 0 0 -e 0.19"
  mlp -t z g6c_training.csv 10000 2 0 0 -e 0.19
  echo " "
  echo "mlp -r z g6c_test.csv"
  mlp -r z g6c_test.csv
  echo "#STOP################################"
  echo " "
done