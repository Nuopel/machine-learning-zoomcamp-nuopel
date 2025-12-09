

## 📺 Session 9.1 – Serverless Deep Learning with AWS Lambda

---

### ✅ Goal

The goal of this session is to learn how to **deploy a trained deep learning model** (specifically a Keras-based image classification model) using **AWS Lambda**, a serverless computing platform. The session shows how to make the trained model available as a web service that can take an input image (such as a photo of pants) and return the predicted class label. It covers model conversion using **TensorFlow Lite** for optimization, building a **Docker container** for deployment, and exposing the service through **API Gateway**, thus enabling lightweight, cost-efficient, and scalable deployment of ML models.

---

### 🧠 Concept

The key concept introduced in this session is **serverless model deployment**, specifically using **AWS Lambda**. Traditional model deployment involves maintaining dedicated servers or containers where the model runs continuously, leading to high costs and complexity. Serverless computing changes this paradigm: functions (like predictions) are run **on-demand**, and the infrastructure is fully managed by AWS. You only pay when your code is executed, which makes it ideal for sporadic or low-traffic applications.

To make models compatible with AWS Lambda's tight resource and cold-start constraints, the session introduces **TensorFlow Lite (TFLite)**. TFLite provides a lightweight, optimized version of TensorFlow models, reducing size and startup time significantly. The deployment is done inside a **Docker container** because Lambda now supports container-based functions (up to 10 GB in size), allowing more flexibility than traditional zipped Lambda functions.

The full deployment pipeline involves:

* Converting a Keras model to TensorFlow Lite
* Writing inference logic that loads and runs the model
* Packaging the logic and the model into a Docker container
* Deploying to AWS Lambda
* Creating an API endpoint via API Gateway to expose the inference service.

---

### 🧠 Key Points Learned Here

1. **Introduction to AWS Lambda for ML Deployment**: AWS Lambda is a serverless service that allows you to run functions without provisioning or managing servers. It is ideal for lightweight, stateless model inference, especially when requests are sporadic and cost efficiency is a concern.

2. **Why TensorFlow Lite (TFLite)**: Regular TensorFlow models are too large and slow to load in Lambda environments. TensorFlow Lite converts the trained model into a compact, optimized format, enabling fast startup and reduced memory usage. This is critical for inference tasks in a serverless setup where functions must load quickly.

3. **Model Conversion Pipeline**: You learn how to take a trained Keras model (e.g., for classifying clothing images) and convert it to TFLite using the `TFLiteConverter`. This process reduces the model's size and prepares it for deployment.

4. **Deployment via Docker Container**: You create a Dockerfile containing the Python environment, inference code, and the TFLite model. This container is then deployed to AWS Lambda, leveraging the recent support for container images up to 10 GB.

5. **Exposing the Model with API Gateway**: Finally, the deployed Lambda function is exposed as a REST API endpoint via API Gateway. This enables any client (e.g., a mobile app or frontend) to send an image URL and receive back a prediction.

6. **Use Case Recap**: The full workflow revolves around a user uploading a photo of clothing (e.g., pants) to a web platform. The backend sends the image to the Lambda function, which returns a classification (e.g., "pants"), and this label is used to categorize the item in the online marketplace.

This session blends machine learning, cloud services, and software engineering into a practical deployment example that is lightweight, production-friendly, and cost-effective.

---

Let me know if you'd like a diagram of the deployment pipeline or a summary table of AWS Lambda vs traditional hosting.




Here is the structured recap of the lesson from **Machine Learning Zoomcamp – Session 9, Lesson 3: TensorFlow Lite**:

---

## 📺 Session 9.3 – Deploying with TensorFlow Lite

---

### ✅ Goal

The objective of this session is to **convert a trained Keras model into TensorFlow Lite (TFLite) format** for lightweight and efficient inference, especially in **resource-constrained environments** like AWS Lambda. TensorFlow Lite allows reducing the model size and dependency footprint, which is essential for fast cold starts, low memory usage, and minimal storage in serverless deployments. The lesson walks through model conversion, inference using the TFLite runtime, and removing dependencies on the full TensorFlow library by replacing image preprocessing logic manually or via helper packages.

---

### 🧠 Concept

The key concept of this lesson is **TensorFlow Lite (TFLite)** – a streamlined version of TensorFlow specifically designed for model inference. Unlike full TensorFlow, which includes training utilities, TFLite is optimized for **deployment and prediction only**, making it faster and smaller. This is crucial in serverless environments like AWS Lambda, where package size and startup latency matter. Full TensorFlow can weigh over 1.7 GB and has a higher memory and loading overhead, which is not ideal for stateless functions.

To leverage TFLite, we need to **convert the trained model** (typically saved in `.h5` or SavedModel format) into `.tflite` using `TFLiteConverter`. Once converted, we use a **TFLite Interpreter** for inference. TensorFlow Lite has a lower-level API, so we need to manually load the model, allocate tensors, set inputs, invoke the model, and retrieve the output.

To avoid importing full TensorFlow just for pre-processing, the lesson explores how to **replicate Keras image preprocessing functions manually** using PIL (Python Imaging Library) and NumPy. As a convenient alternative, the `keras-image-helper` package provides a high-level abstraction for loading and pre-processing images compatible with known architectures like Xception.

Finally, instead of using the full TensorFlow installation (which includes TFLite), we install **`tflite-runtime`** separately. This dramatically reduces the deployment footprint.

---

### 🧠 Key Points Learned Here

* **Why TensorFlow Lite**: Full TensorFlow is large and slow to load, which affects serverless function performance and cost. TensorFlow Lite focuses only on inference, reducing size and latency.

* **Conversion Process**: A Keras `.h5` model is converted to TFLite format using `TFLiteConverter.from_keras_model(model).convert()`. This yields a `.tflite` binary that can be saved and reused.

* **TFLite Inference Workflow**:

  * Load the model using `Interpreter(model_path)`
  * Call `.allocate_tensors()` to prepare memory
  * Retrieve input/output tensor indexes with `get_input_details()` and `get_output_details()`
  * Set input via `.set_tensor(input_index, input_data)`
  * Run inference via `.invoke()`
  * Retrieve results via `.get_tensor(output_index)`

* **Replacing TensorFlow Preprocessing**: Instead of using `tf.keras.preprocessing.image.load_img` and `tf.keras.applications.xception.preprocess_input`, we manually resize and normalize the image using PIL and NumPy to avoid TensorFlow dependency.

* **Using `keras-image-helper`**: A simple library that abstracts image loading and preprocessing tailored to specific Keras architectures (e.g., Xception), simplifying the workflow and removing the need for manual image manipulation.

* **Installing `tflite-runtime`**: By using `pip install tflite-runtime`, we can run inference with just the TFLite interpreter, completely avoiding the large TensorFlow installation, which is critical for AWS Lambda environments.

---

### 🛠️ Tasks

1. ✅ **Downloaded pre-trained `.h5` model** from the course release.
2. ✅ **Loaded the model** using `tf.keras.models.load_model`.
3. ✅ **Loaded and preprocessed image** using `tf.keras.preprocessing.image` and `tf.keras.applications.xception.preprocess_input`.
4. ✅ **Converted the model** to TFLite format using `TFLiteConverter`.
5. ✅ **Saved the converted model** as `.tflite`.
6. ✅ **Loaded the `.tflite` model** using `tflite.Interpreter`.
7. ✅ **Allocated tensors**, set input, invoked the model, and retrieved output.
8. ✅ **Replaced TensorFlow-based image preprocessing** with manual PIL/NumPy equivalents.
9. ✅ **Tested inference with TensorFlow Lite only** (no full TensorFlow installed).
10. ✅ **Introduced `keras-image-helper`** for simpler image preprocessing.
11. ✅ **Installed and tested `tflite-runtime`** as a drop-in replacement for full TensorFlow.

---


Here’s the structured recap of **Machine Learning Zoomcamp – Session 9, Lesson 5: Dockerizing a Model for AWS Lambda**:

---

## 📺 Session 9.5 – Packaging a TensorFlow Lite Model with Docker for AWS Lambda

---

### ✅ Goal

The goal of this session is to **containerize a TensorFlow Lite inference service using Docker** in order to deploy it on AWS Lambda. This involves creating a `Dockerfile` that installs necessary dependencies, copies the Python inference script and model, and sets the correct AWS Lambda handler. The resulting Docker image is then tested locally with a sample prediction request to ensure the inference works correctly before deploying it to the cloud.

---

### 🧠 Concept

This lesson focuses on the **Dockerization of a serverless deep learning model** using AWS Lambda’s support for **container-based functions**. Traditionally, AWS Lambda functions are deployed using ZIP archives, with a 50MB limit on code size. With the newer approach, you can package your function as a **Docker container (up to 10 GB)**, enabling inclusion of custom binaries, models, and dependencies.

The base image used comes from AWS’s **public ECR (Elastic Container Registry)**, specifically tailored for Python on Lambda. To ensure compatibility with AWS Lambda's Amazon Linux runtime, the lesson emphasizes the importance of using **precompiled binaries** for critical libraries like `tflite-runtime`, which are platform-dependent. The inference logic, originally written in a Jupyter notebook, is transferred into a Python script, which is included in the Docker container.

Testing is done by running the Docker container locally and sending a POST request using the `requests` library to simulate how AWS will call the function. During this, common pitfalls such as serialization errors from NumPy arrays are addressed, and a workaround is implemented by converting prediction outputs to native Python types.

---

### 🧠 Key Points Learned Here

* **AWS Lambda + Docker**: AWS now supports deploying Lambda functions as Docker containers, providing more flexibility for machine learning models and binary dependencies that wouldn’t fit or work within ZIP-based deployments.

* **Base Image Selection**: Use official AWS Lambda base images (e.g., `public.ecr.aws/lambda/python:3.8`) to ensure compatibility with the Lambda runtime.

* **Platform Compatibility**: Standard `tflite-runtime` wheels may be incompatible with Amazon Linux (used by Lambda). It’s important to install a **precompiled wheel** that matches the target environment’s glibc version. A compatible wheel was hosted on GitHub and installed directly via URL.

* **Dockerfile Structure**:

  * Install required packages (`keras-image-helper`, `tflite-runtime` via GitHub-hosted wheel).
  * Copy the `.tflite` model and the handler script (`lambda_function.py`).
  * Define the Lambda handler using `CMD ["lambda_function.handler"]`.

* **Inference Script Adjustments**:

  * Predictions from NumPy arrays are not JSON-serializable; they must be converted to standard Python floats/lists.
  * The inference endpoint uses AWS’s specific path: `POST /2015-03-31/functions/function/invocations`.

* **Local Testing**:

  * Docker container is run locally exposing port `8080`.
  * A test script sends an HTTP POST request with a sample image URL.
  * Successful JSON response with predictions confirms the image is working.

---

### 🛠️ Tasks

1. ✅ Converted Jupyter notebook into a clean Python inference script.
2. ✅ Chose AWS Lambda-compatible base Docker image (`python:3.8`).
3. ✅ Wrote a `Dockerfile`:

   * Installed `keras-image-helper`.
   * Installed `tflite-runtime` from a precompiled wheel.
   * Copied `.tflite` model and handler script.
   * Declared the function handler with `CMD`.
4. ✅ Built the Docker image locally (`docker build -t clothing-model .`).
5. ✅ Ran the container (`docker run -p 8080:8080 clothing-model`).
6. ✅ Wrote a Python test script using `requests` to simulate AWS invocation.
7. ✅ Fixed NumPy serialization error by converting predictions to native Python types.
8. ✅ Validated that the container returns predictions successfully from the TFLite model.

---

Let me know if you'd like a copy of the full `Dockerfile`, handler script template, or deployment commands for AWS CLI/ECR.

Here is the structured recap of **Machine Learning Zoomcamp – Session 9, Lesson 6: Creating a Lambda Function**:

---

## 📺 Session 9.6 – Deploying a Dockerized Model to AWS Lambda

---

### ✅ Goal

The goal of this session is to **deploy a machine learning inference service as an AWS Lambda function using a Docker container image**. After testing the Docker image locally, the session guides you through publishing the image to **Amazon ECR (Elastic Container Registry)** and creating a Lambda function that uses this container. The function is then configured, tested, and analyzed for runtime behavior and cost. This allows you to serve deep learning models in a **serverless, scalable, and infrastructure-free** manner.

---

### 🧠 Concept

This lesson builds on the concept of **serverless ML deployment** using **containerized Lambda functions**. Instead of deploying code via ZIP archives, AWS now allows entire **Docker containers** to be used as function packages. These containers must be hosted on **Amazon ECR**, which integrates natively with Lambda.

The session explains how to:

* Create and authenticate to an ECR repository using AWS CLI.
* Push a prebuilt Docker image that contains a TensorFlow Lite model and Python inference script.
* Configure and deploy the Lambda function using the AWS Console.
* Adjust runtime settings like **memory allocation** and **timeout** to improve performance.
* Handle cold starts and test the function by sending real inference requests.
* Understand the **pricing model** for AWS Lambda based on memory and execution duration.

This architecture enables flexible, stateless, and cost-effective deployment of ML inference logic, especially for **low-throughput or bursty workloads**.

---

### 🧠 Key Points Learned Here

* **Amazon ECR (Elastic Container Registry)** is the hosting platform for Docker images used in Lambda. Images must be pushed there before they can be referenced in Lambda.

* **Authentication & Push Flow**:

  * Use `aws ecr create-repository` to make a new registry.
  * Authenticate Docker with ECR using `aws ecr get-login-password`.
  * Tag the local image with the ECR URI and push using `docker push`.

* **Creating the Lambda Function**:

  * Choose **"Container image"** as the function source.
  * Provide the ECR image URI (or use the UI to browse).
  * AWS uses a **digest** reference behind the scenes.

* **Runtime Configuration**:

  * Default **timeout (3s)** was too short; increased to **30 seconds**.
  * Default **memory (128MB)** was increased to **1GB** to speed up execution.
  * Cold starts (first invocation) are slower (\~7–8s), warm invocations are faster (\~2s).

* **Cost Estimation**:

  * AWS charges based on **duration x memory usage**.
  * Estimated **\$0.33 for 10,000 inferences**, or **\$33 for 1M** with 1GB RAM and 2s runtime.
  * Using **ARM architecture** can reduce cost by \~20% in some cases.
  * Good cost/performance for **testing or moderate loads**, but can become costly at scale.

* **Limitations**:

  * No code preview for container-based Lambda.
  * Manual management of platform compatibility (e.g., `tflite-runtime` must be compiled for Amazon Linux).

* **Use Case Fit**:

  * Ideal for **prototyping and experimentation**.
  * Not recommended for **high-volume production inference** without considering pricing implications.

---

### 🛠️ Tasks

1. ✅ Verified Docker image worked locally.
2. ✅ Installed/configured **AWS CLI**.
3. ✅ Created ECR repository:
   `aws ecr create-repository --repository-name clothing-tflite-images`
4. ✅ Logged in to ECR using CLI with secure password handling via `aws ecr get-login-password | docker login`.
5. ✅ Tagged the local Docker image using:
   `docker tag clothing-model <account-id>.dkr.ecr.<region>.amazonaws.com/clothing-model:v1`
6. ✅ Pushed image to ECR:
   `docker push <ecr-uri>`
7. ✅ Created Lambda function via AWS Console (container image source).
8. ✅ Configured Lambda runtime (timeout = 30s, memory = 1024MB).
9. ✅ Created test event using a sample image URL and invoked the function.
10. ✅ Observed **cold start** vs. **warm start** durations.
11. ✅ Used **AWS pricing calculator** to estimate per-inference cost.
12. ✅ Noted **architecture-based cost difference** (ARM vs x86).
13. ✅ Concluded that Lambda is great for testing and low-volume inference, but not ideal for heavy workloads.

---


Here is the structured recap of **Machine Learning Zoomcamp – Session 9, Lesson 7: Exposing a Lambda Function as a Web Service**:

---

## 📺 Session 9.7 – Exposing Lambda with API Gateway

---

### ✅ Goal

The goal of this session is to **expose an AWS Lambda function as a public web service** using **API Gateway**. After successfully deploying the machine learning inference function using a Docker container on AWS Lambda, the next step is to make it accessible externally through an HTTP endpoint. This allows any application or user to send a request to the model via a RESTful interface (e.g., POST request with an image URL) and receive predictions in response. The lesson covers setting up a REST API with a `/predict` endpoint, linking it to the Lambda function, testing it, and discussing access control considerations.

---

### 🧠 Concept

This session introduces **AWS API Gateway** as the tool to transform a Lambda function into a web-accessible API. In a serverless architecture, Lambda functions are powerful but require a front-facing mechanism to receive external requests—this is where API Gateway comes in.

API Gateway acts as a **proxy layer** between the external world and AWS services. It can:

* Route incoming HTTP(S) requests (e.g., `POST /predict`)
* Trigger Lambda functions with the request payload
* Format and return the Lambda function’s response back to the client

The REST API is built using standard HTTP principles (resources, methods, stages). In this lesson, a `/predict` resource is defined with a **POST method**, which forwards the incoming JSON payload to the Lambda function. A stage named `test` is created to make the API publicly accessible, generating a usable endpoint URL.

Security is an important consideration—by default, the endpoint is public, which is acceptable for learning purposes but **must be restricted in production environments** using IAM policies, API keys, or authorization tokens.

---

### 🧠 Key Points Learned Here

* **API Gateway Setup**:

  * A new **REST API** was created with the name `clothing-classification`.
  * A **resource** `/predict` was defined, mimicking common ML serving conventions.
  * A **POST method** was linked to the deployed Lambda function.

* **Integration with Lambda**:

  * During method creation, API Gateway is granted permission to invoke the Lambda function.
  * The request payload (a JSON with an image URL) is forwarded to the Lambda function.
  * The function returns prediction results, which API Gateway relays to the caller.

* **Testing and Deployment**:

  * The endpoint was tested directly from the AWS Console using a sample JSON payload.
  * After verifying functionality, the API was **deployed to a stage** (`test`), producing a public URL.
  * External testing via Python (`requests.post`) confirmed the end-to-end flow: `API Gateway → Lambda → Response`.

* **Security Note**:

  * By default, the endpoint is **public and unauthenticated**, which is suitable for learning but not for production.
  * In production, one must configure **access control** mechanisms (e.g., usage plans, tokens, VPC link, or IAM roles).

* **Outcome**:

  * The deployed model is now **accessible as a REST API** and can be queried from anywhere with a POST request.

---

### 🛠️ Tasks

1. ✅ Navigated to **API Gateway** in AWS Console.
2. ✅ Created a new **REST API** named `clothing-classification`.
3. ✅ Defined a new **resource** `/predict`.
4. ✅ Added a **POST method** linked to the deployed Lambda function.
5. ✅ Accepted IAM permissions prompt to allow invocation from API Gateway.
6. ✅ Used the **test interface** within API Gateway to send a sample JSON payload (`{"url": "<image_url>"}`).
7. ✅ Verified successful responses and latency measurements.
8. ✅ **Deployed the API** to a new stage (`test`) to make it publicly accessible.
9. ✅ Updated the **Python test script** to point to the new API Gateway URL.
10. ✅ Tested end-to-end web request from client → API Gateway → Lambda → prediction response.
11. ✅ Highlighted **security risks** and reminded users to restrict access in production environments.

---


## 📺 Session 9.8 – Summary: Serverless Deep Learning with AWS Lambda

---

### ✅ Goal

The goal of this session is to **recap the process of deploying deep learning models in a serverless fashion using AWS Lambda**, highlighting key concepts, tools, benefits, limitations, and real-world applicability. The focus is on consolidating what was learned: packaging a model into a Docker image, converting it with TensorFlow Lite, testing it locally, deploying to Lambda via ECR, and exposing it with API Gateway. The session emphasizes why this approach is effective for **low-traffic, cost-efficient, and infrastructure-free inference** services, especially for personal or experimental ML projects.

---

### 🧠 Concept

This lesson reinforces the idea of **serverless deployment using AWS Lambda**, where you don't manage or provision servers yourself. Instead, you package your inference logic (e.g., TensorFlow Lite model) into a container image and push it to **Amazon ECR**. Then, you create a Lambda function that uses this image to serve predictions on demand. The key advantage of this model is that you **only pay when the function is invoked**, making it ideal for applications with **sporadic usage**.

One technical innovation introduced was the use of **TensorFlow Lite**, which drastically reduces model size and memory consumption—crucial in constrained environments like Lambda. However, it also requires a more verbose setup compared to plain `model.predict()` with Keras. Despite this, using TFLite results in container images that are **over 100x smaller** than those using full TensorFlow, enabling faster cold starts and lower cloud costs.

The session also opens the door to **general-purpose model deployment** on Lambda—whether for deep learning (e.g., CNNs, transformers) or traditional ML (e.g., XGBoost, scikit-learn models)—and encourages learners to experiment and apply these techniques to their own capstone or real-world projects.

---

### 🧠 Key Points Learned Here

* **Lambda is Ideal for Low-Traffic ML Inference**: You only pay for execution time and allocated memory, making it very cost-effective for personal use or infrequent workloads.

* **Container-Based Deployment**:

  * By Dockerizing your model, you ensure full control over the runtime environment.
  * Local testing with Docker means fewer surprises at deployment.

* **TensorFlow Lite for Inference**:

  * TFLite is lightweight (≈2–3 MB), optimized for inference only, and enables faster loading and smaller Docker images.
  * In contrast, full TensorFlow is over 200–500 MB when packaged, making it inefficient for Lambda.

* **Workflow Recap**:

  * Convert model to TFLite.
  * Write Lambda-compatible inference script.
  * Package using Docker.
  * Push to ECR and link to Lambda.
  * Expose via API Gateway.

* **Flexibility**:

  * You’re not limited to AWS—similar functionality is available on **Google Cloud Functions**, **Azure Functions**, and other platforms.
  * You can deploy not only neural networks, but also traditional ML models using libraries like scikit-learn or XGBoost.

* **Performance Trade-offs**:

  * Cold starts (especially the first request) are slower due to model and environment loading.
  * Warm invocations are fast and efficient.

* **Future Exploration**:

  * Kubernetes-based deployment offers better scalability and control, which will be covered in the next session.

---

### 🛠️ Tasks

1. ✅ Converted Keras model to **TensorFlow Lite** format.
2. ✅ Created a Python inference script compatible with AWS Lambda.
3. ✅ Packaged the code and model into a **Docker image**.
4. ✅ Tested the image **locally** using Docker.
5. ✅ Created an **Amazon ECR repository** and pushed the Docker image.
6. ✅ Deployed the image as an **AWS Lambda container-based function**.
7. ✅ Adjusted **timeout and memory** settings for optimal performance.
8. ✅ Invoked and tested the function, observing **cold vs. warm start behavior**.
9. ✅ Discussed **Lambda pricing**, including per-inference costs and memory trade-offs.
10. ✅ Explored future directions (API Gateway for exposure, Kubernetes as alternative, traditional ML support).

---

Let me know if you'd like a complete checklist or deployment template for serverless ML projects.
