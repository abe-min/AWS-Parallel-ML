# Parallel Machine Learning in AWS

### GitHub Link: https://github.com/abe-min/CS-643-Programming-Assignment-2
### DokerHub Link: https://hub.docker.com/r/abemindaye/progass2 

## Assignment Information
The purpose of this individual assignment is to learn how to develop parallel machine learning (ML) applications in Amazon AWS cloud platform. Specifically, you will learn: (1) how to use Apache Spark to train an ML model in parallel on multiple EC2 instances; (2) how to use Spark’s MLlib to develop and use an ML model in the cloud; (3) How to use Docker to create a container for your ML model to simplify model deployment.


# Assignment Implementation
## Launching AWS Elastic Map Reduce 
1. Login to the Student AWS Learner account through the link sent by the TA.
![Learner Lab Home](https://github.com/abe-min/CS-643-Programming-Assignment-2/blob/main/files/Learner_Lab_Home.PNG?raw=true "AWS Student Learner Lab Home")
2. Click "Start Lab" and navigate to the AWS Management Console.
3. From the AWS Console, select EMR
4. Using the EMR console, configure a cluster with 4 EC2 instances 
![EMR Console](https://github.com/abe-min/CS-643-Programming-Assignment-2/blob/main/files/EMR_Config.png?raw=true "EMR Console Configuration")
	<br>The cluster will be comprised of:
		<br>1x Primary node 
		<br>4x Core nodes 
		<br>1x Task node
5. Create S3 bucket which will be used for saving the training outresult (model). S3 bucket will also be used to hold the training and validating dataset files. 
![S3 Storage](https://github.com/abe-min/CS-643-Programming-Assignment-2/blob/main/files/S3_Bucket.png?raw=true "S3 Storage")

## Connect to the Primary Node of the EMR Cluster  
1. Navigate to the list of running clusters on the EMR console. If the cluster initialized successfully, we should see "Ready" next to cluster "wine" 
2. Find the public addr of the Primary node in the "wine" cluster and SSH to it using the certefticate keys. `ssh -i April_23_2023.cer hadoop@ec2-54-242-71-20.compute-1.amazonaws.com`
3. Once successfully connected to the primary node, it should looks as follows.
![SSH Terminal](https://github.com/abe-min/CS-643-Programming-Assignment-2/blob/main/files/EMR_primary_terminal.png?raw=true "SSH Terminal")

## Training the Model
1. Transfer the training script into the primary node. Here I am going to create a train.py script in the node with `vim train.py` and copy and paste the training script from my local machine. This script will do the following:
<br>a. Read the TrainingDataset.csv into Spark
<br>b. train a model with forest training classifier with 10 classes 
<br>c. Output the model to s3 storage location
2. Once the script is ready, we can run it with the command `spark-submit train.py` 
![Spark Submit](https://github.com/abe-min/CS-643-Programming-Assignment-2/blob/main/files/Spark-Submit.png?raw=true "Spark Submit")

3. When the job is completed, it will output the model to the s3 bucket: https://wine-emr-bucket.s3.amazonaws.com/output/
![Saved Model](https://github.com/abe-min/CS-643-Programming-Assignment-2/blob/main/files/trained_model.png?raw=true "Saved Model")

## Testing/Validating the Model
1. Once the model is saved, we can load it into a Prediction.py script and test the model.
2. The Prediction.py script will import the trained model saved in s3 and use it in making preditions using the ValidationDataset.csv. 

# Prediction Application
## Running Prediction Application with Docker on AWS 
1. Launch an AWS EC2 instance
2. Docker pull image with command `docker pull abemindaye/progass2:latest`
3. Run the Prediction application with command `docker run -p 4000:80 abemindaye/progass2:latest`. This will run the Prediction.py application with the ValidationDataset.csv in the docker image as input. 
4. To test with another CSV(ie TestDataset.csv), copy it to the working directory in the host EC2 instance and rename it to ValidationDataset.csv. After that, run the following command, which will replace the ValidationDataset.csv file in the docker image with the new one from the host. `docker run -p 4000:80 -v ValidationDataset.csv:/winepredict/ abemindaye/progass2:latest`
## Running Prediction Application without Docker on AWS
1. Launch an AWS EC2 instance
2. Install the following: **findspark**, **pandas**, **sklearn**: `python3 -m pip install scikit-learn findspark pandas pyspark`
4. `git clone https://github.com/abe-min/CS-643-Programming-Assignment-2` (if git is not installed on the EC2 instance being used, install with: `sudo yum install git -y`)
5. `cd CS-643-Programming-Assignment-2`
6. Run the Prediction.py script and pass the dataset as an argument. For example: `python3 Prediction.py training_files/ValidationDataset.csv`
7. Expected Prediction.py output: 
![Prediction Output](https://github.com/abe-min/CS-643-Programming-Assignment-2/blob/main/files/prediction_output.png?raw=true "python Prediction Output")
