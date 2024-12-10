#1. Set the base image to python 3.11
FROM python:3.11 

#2. Set the working directory inside a container 
WORKDIR /app_dir 

#3. Copy the req. txt to the working directory 
COPY ./requirements.txt /app_dir/ 

#4. Install the python dependencies 
RUN pip install -r requirements.txt 

#5. Copy the application code into the container 
COPY . /app_dir/ 

#5.2 Set the flask environment variables 
ENV FLASK_APP=app.py 

#6. Expose the port that the application will use 
EXPOSE 5000 

#7. Define the default command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

