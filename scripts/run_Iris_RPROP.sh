for i in {1..25}
do
  echo "$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i."
  echo "mlp -c z 3 2 2 1"
  mlp -c z 3 2 2 1
  echo " "
  echo "mlp -t z Iris_training.csv 10000 2 0 0"
  mlp -t z Iris_training.csv 10000 2 0 0
  echo " "
  echo "mlp -r z Iris_test.csv"
  mlp -r z Iris_test.csv
  echo "$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i.$i."
  echo " "
done