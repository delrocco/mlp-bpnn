for i in {1..25}
do
  echo "#START###############################"
  echo "mlp -c z 3 2 2 4"
  mlp -c z 3 2 2 4
  echo " "
  echo "mlp -t z g4c_training.csv 12000 2 0 0 -e 0.26"
  mlp -t z g4c_training.csv 12000 2 0 0 -e 0.26
  echo " "
  echo "mlp -r z g4c_test.csv"
  mlp -r z g4c_test.csv
  echo "#STOP################################"
  echo " "
done