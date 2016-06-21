for i in {1..25}
do
  echo "#START###############################"
  echo "mlp -c z 4 2 9 4"
  mlp -c z 4 2 9 4
  echo " "
  echo "mlp -t z g4c_training.csv 3000 1 0.04 0.75 -e 0.26"
  mlp -t z g4c_training.csv 3000 1 0.04 0.75 -e 0.26
  echo " "
  echo "mlp -r z g4c_test.csv"
  mlp -r z g4c_test.csv
  echo "#STOP################################"
  echo " "
done