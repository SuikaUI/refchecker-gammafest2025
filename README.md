# 🏆 Automatic Inter-Paper Citation Prediction System

## Competition Results
**RANK 1 WINNER** with **Most Creative Approach** out of **100+ teams** (approx. **300+ participants**) 

**Final Score:** 0.616 MCC (Private Leaderboard)

---

## Team Members

| Name | GitHub Profile |
|------|----------------|
| **Rahardi Salim** | [@RahardiSalim](https://github.com/RahardiSalim) |
| **Christian Yudistira Hermawan** | [@Nadekoooo](https://github.com/Nadekoooo) |
| **Vincent Davis Leonard Tjoeng** | [@Vincent-Davis](https://github.com/vincent-davis) |

---

## Project Overview

This project presents an innovative approach to automatic inter-paper citation prediction by combining **Document Embedding**, **Chunk-based Features**, and **Metadata Aggregation**. Our multi-view learning system significantly outperforms traditional approaches by integrating three complementary perspectives of document relationships.

### Problem Statement

With the exponential growth of scientific literature, researchers struggle to find relevant references using traditional methods. Citation networks are retrospective, and keyword-based approaches fail to capture complex semantic relationships, causing potentially valuable papers to be overlooked.

### Our Solution

We developed a comprehensive citation prediction system that combines:
- **Global Textual View**: Document-level embeddings using SPECTER
- **Local Textual View**: Chunk-level similarity analysis using all-MiniLM-L6-v2
- **Non-Textual View**: Rich metadata features and bibliographic context

---

## Web Demo

We've built an interactive web application to demonstrate our citation prediction system!

### Features:
- **Real-time Prediction**: Upload papers to get instant citation predictions
- **Interactive Analysis**: Explore similarity scores and feature contributions
- **User-friendly Interface**: Simple upload and analysis workflow

### Running the Demo:
```bash
cd app
pip install -r requirements.txt
python manage.py runserver
```

---

## Results & Performance

### 🏆 Competition Performance
- **Private Leaderboard Score**: 0.616 MCC
- **Significant improvement** over baseline approaches:
  - Document-level only: ~0.372 MCC
  - Document + Chunk (no metadata): ~0.510 MCC
  - **Our full system**: 0.616 MCC

### Key Insights:
- **Most Important Features**: Global similarity differences and local chunk variance
- **Finding**: Citation patterns involve complex relationships beyond pure semantic similarity

---

## Project Structure

```
├── app/                   # Web application (Django)
├── dataset/               # Training and test data
├── docs/                  # Presentation and report
└── notebooks/             # Research notebooks
```

---

## 🏅 Achievement Highlights

- **1st Place** in Citation Prediction Competition  
- **Most Creative Approach** Award  
- **60%+ improvement** over baseline methods  
- **Novel multi-view learning architecture**  
- **Interactive web demo** for practical use  

---

## License

This project is licensed under the MIT License.  
© 2025 Gammafest IPB.
