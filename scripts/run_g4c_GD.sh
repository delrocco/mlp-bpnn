for i in {1..25}
do
  echo "#START###############################"
  echo "mlp -c z 4 2 9 4"
  mlp -c z 4 2 9 4
  echo " "
  echo "mlp -t z g4c_training.csv 2000 0 0.85 0 -e 0.22"
  mlp -t z g4c_training.csv 2000 0 0.85 0 -e 0.22
  echo " "
  echo "mlp -r z g4c_test.csv"
  mlp -r z g4c_test.csv
  echo "#STOP################################"
  echo " "
done