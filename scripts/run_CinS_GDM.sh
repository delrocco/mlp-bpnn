for i in {1..25}
do
  echo "$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i."
  echo "mlp -c z 3 2 16 1"
  mlp -c z 3 2 16 1
  echo " "
  echo "mlp -t z CinS_training.csv 5000 1 0.06 0.5"
  mlp -t z CinS_training.csv 5000 1 0.06 0.5
  echo " "
  echo "mlp -r z CinS_test.csv"
  mlp -r z CinS_test.csv
  echo "$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i."
  echo " "
done