# 📧 Spam Email Detection System

An end-to-end Machine Learning application that detects spam emails using **TF-IDF** and **Support Vector Machine (SVM)**. The project is built with **FastAPI**, **Streamlit**, **Docker**, and deployed on **Amazon EKS (Kubernetes)** using **Amazon ECR** and **GitHub Actions CI/CD**.

---

## 🚀 Features

- Spam Email Classification using SVM
- TF-IDF Text Vectorization
- Single & Batch Email Prediction
- FastAPI REST API
- Streamlit Web Interface
- Dockerized Application
- GitHub Actions CI/CD
- Amazon ECR Integration
- Amazon EKS Deployment
- AWS Application Load Balancer
- Kubernetes Horizontal Pod Autoscaler (HPA)
- Health Check Endpoint
- Prometheus Metrics Endpoint
- Request Logging

---

## 🛠️ Tech Stack

**Machine Learning**
- Python
- Scikit-learn
- TF-IDF
- Support Vector Machine (SVM)

**Backend**
- FastAPI
- Pydantic
- Uvicorn

**Frontend**
- Streamlit

**DevOps & Cloud**
- Docker
- Kubernetes
- Amazon EKS
- Amazon ECR
- AWS Application Load Balancer
- GitHub Actions
- Prometheus Client

---

## ☁️ Architecture

```text
User
   │
   ▼
Streamlit Frontend
   │
HTTP Request
   │
AWS Application Load Balancer
   │
Amazon EKS
   │
FastAPI
   │
TF-IDF + SVM Model
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health Check |
| GET | `/metrics` | Prometheus Metrics |
| POST | `/single_predict` | Predict Single Email |
| POST | `/batch_predict` | Predict Multiple Emails |

---

## 🚀 Deployment

- Docker
- GitHub Actions CI/CD
- Amazon ECR
- Amazon EKS
- Kubernetes Ingress
- Horizontal Pod Autoscaler (HPA)

---



## 👨‍💻 Author

**Robin Kumar**

Machine Learning • MLOps • Cloud Computing
