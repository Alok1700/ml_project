# 🧒 Predictive Forecasting of Care Load & Placement Demand

**U.S. Department of Health and Human Services — Unaccompanied Alien Children (UAC) Program**

A time-series forecasting system that predicts how many children will be in HHS care over the next 7–60 days, estimates discharge (sponsor placement) demand, and provides early-warning signals for capacity stress.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [Features](#-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Dashboard](#-dashboard)
- [Key Performance Indicators](#-key-performance-indicators)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

The UAC Program operates in a high-uncertainty environment where sudden changes in border activity, policy enforcement, or humanitarian crises can rapidly increase the number of children entering federal care.

While descriptive analytics explain **what has already happened**, HHS decision-makers require **forward-looking intelligence** to answer:

- How many children will be under HHS care in the coming days or weeks?
- Will discharge capacity be sufficient to offset incoming transfers?
- When should shelters, medical staff, and caseworkers be scaled up in advance?

This project introduces **predictive modeling** to enable proactive — rather than reactive — healthcare and child-welfare planning.

---

## ❗ The Problem

Despite having high-quality daily time-series data, the UAC Program currently lacks:

- ❌ Short-term forecasts of children in HHS care
- ❌ Predictive estimates of discharge (placement) demand
- ❌ Early-warning indicators of upcoming capacity stress

**Consequences:**
- 🏚️ Overcrowding risk
- 😓 Staff burnout
- ⏳ Increased length of stay for children

---

## 💡 The Solution

A complete forecasting pipeline that:

1. **Forecasts** the number of children in HHS care (7–60 days ahead)
2. **Estimates** the future imbalance between intake and exits
3. **Predicts** short-term discharge (sponsor placement) demand
4. **Flags** capacity stress before it becomes a crisis
5. **Quantifies** uncertainty with confidence intervals

---

## ✨ Features

- 📈 **Multi-horizon forecasting** (7, 14, 30, 60 days)
- 🤖 **6 forecasting models** — from baselines to machine learning
- 🔍 **Walk-forward validation** for realistic evaluation
- 📊 **Interactive Streamlit dashboard** with 4 analytical tabs
- ⚠️ **Early-warning system** based on net-pressure signals
- 📉 **Confidence intervals** for honest uncertainty quantification
- 🧮 **Feature engineering** with lags, rolling stats, flow signals, and calendar effects
- 📋 **Comprehensive evaluation** with MAE, RMSE, MAPE, and stability index

---

## 📊 Dataset

**Source:** HHS UAC Program daily operational records
**Period:** January 2023 – December 2025 (~3 years)
**Frequency:** Daily (with occasional gaps on weekends/holidays)

| Column | Description |
|--------|-------------|
| `Date` | Reporting date |
| `Children apprehended and placed in CBP custody` | Daily intake volume |
| `Children in CBP custody` | Active CBP care load |
| `Children transferred out of CBP custody` | Flow into HHS system |
| `Children in HHS Care` | Active HHS care load ⭐ (primary target) |
| `Children discharged from HHS Care` | Successful sponsor placements |

**Data challenges handled:**
- Irregular reporting (missing weekends/holidays) → time-based interpolation
- Comma-formatted numbers → parsed to numeric
- Trailing empty rows → removed
- Structural breaks (e.g., Jan 2025 surge) → documented and handled

---

## 🔬 Methodology

### 1. Time-Series Preparation
- Convert `Date` to datetime index
- Ensure daily continuity via reindexing
- Interpolate missing values
- Decompose into trend, seasonality, and residuals

### 2. Feature Engineering
| Feature Type | Details |
|--------------|---------|
| **Lag features** | t-1, t-7, t-14 |
| **Rolling stats** | 7-day and 14-day mean & std |
| **Flow signals** | Net pressure = Transfers − Discharges |
| **Calendar effects** | Day of week, month, quarter, weekend flag |

### 3. Train–Test Strategy
- ✅ Strict time-based splits (no random sampling)
- ✅ Walk-forward validation
- ✅ Multi-horizon evaluation

### 4. Forecasting Models

| Category | Models |
|----------|--------|
| **Baseline** | Naïve persistence, Moving average |
| **Statistical** | ARIMA / SARIMA, Exponential Smoothing |
| **Machine Learning** | Random Forest, Gradient Boosting |

### 5. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **MAE** | Absolute forecast accuracy |
| **RMSE** | Penalizes large errors |
| **MAPE** | Relative error understanding |
| **Horizon Error** | Short vs medium-term reliability |
| **Stability Index** | Consistency across folds |

---

## 🏆 Results

- **Gradient Boosting** achieves the best performance for 7–30 day horizons
- **ARIMA** & **Exponential Smoothing** remain competitive for 1–7 day horizons
- **Naïve baseline** is surprisingly hard to beat at 1–3 days
- **Confidence intervals:** ±10% at 14 days, ±25% at 60 days
- **Early-warning signal:** Net pressure > +10/day for 7 days reliably precedes care load rise

**Key insight:** No purely historical model can predict structural breaks (e.g., Jan 2025 surge). Scenario planning is recommended alongside statistical forecasts.

---

## ⚙️ Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/uac-care-load-forecasting.git
cd uac-care-load-forecasting
