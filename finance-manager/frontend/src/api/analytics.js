import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
});

// Add token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const fetchSpendingPatterns = async () => {
  const { data } = await api.get('/analytics/spending-patterns');
  return data;
};

export const fetchMonthlyPrediction = async () => {
  const { data } = await api.get('/analytics/monthly-prediction');
  return data;
};

export const fetchUnusualTransactions = async () => {
  const { data } = await api.get('/analytics/unusual-transactions');
  return data;
};
