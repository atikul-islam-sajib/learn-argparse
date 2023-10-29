
## Using the Iris Classifier

To use the Iris Classifier, follow these steps:

1. Open a command prompt or terminal.

2. Navigate to the directory containing the `iris_classifier.py` script using the `cd` command. For example:

   ```shell
   cd C:\Users\atiku\OneDrive\Desktop\classifier_classifier\classifier\command_line
   ```

3. To train the model with the Iris dataset, use the following command:

   ```shell
   python iris_classifier.py --dataset Iris.csv --split 0.25 --train
   ```

   This command will train the model, and you should see the message "Model Trained Successfully" upon successful completion.

4. To evaluate the trained model, use the following command:

   ```shell
   python iris_classifier.py --evaluate
   ```

   This command will provide accuracy, precision, recall, and F1 score metrics for the classifier.

Example Output:
```shell
ACCURACY # 1.0
PRECISION # 1.0
RECALL # 1.0
F1_SCORE # 1.0
```
