for i in {1..25}
do
  echo "$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i."
  echo "mlp -c z 3 2 3 1"
  mlp -c z 3 2 3 1
  echo " "
  echo "mlp -t z CinS_training.csv 10000 2 0 0"
  mlp -t z CinS_training.csv 10000 2 0 0
  echo " "
  echo "mlp -r z CinS_test.csv"
  mlp -r z CinS_test.csv
  echo "$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i."
  echo " "
done