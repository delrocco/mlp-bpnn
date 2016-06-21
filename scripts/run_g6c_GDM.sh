for i in {1..25}
do
  echo "#START###############################"
  echo "mlp -c z 3 2 7 6"
  mlp -c z 3 2 7 6
  echo " "
  echo "mlp -t z g6c_training.csv 10000 1 0.05 0.85 -e 0.155"
  mlp -t z g6c_training.csv 10000 1 0.05 0.85 -e 0.155
  echo " "
  echo "mlp -r z g6c_test.csv"
  mlp -r z g6c_test.csv
  echo "#STOP################################"
  echo " "
done