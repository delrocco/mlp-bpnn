for i in {1..25}
do
  echo "$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i."
  echo "mlp -c z 3 2 4 1"
  mlp -c z 3 2 4 1
  echo " "
  echo "mlp -t z Iris_training.csv 5000 1 0.06 0.85"
  mlp -t z Iris_training.csv 5000 1 0.06 0.85
  echo " "
  echo "mlp -r z Iris_test.csv"
  mlp -r z Iris_test.csv
  echo "$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i."
  echo " "
done