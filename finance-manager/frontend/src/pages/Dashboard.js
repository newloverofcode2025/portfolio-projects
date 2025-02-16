import React from 'react';
import { Grid, Paper, Typography } from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { Bar, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';

import { fetchSpendingPatterns, fetchMonthlyPrediction } from '../api/analytics';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

export default function Dashboard() {
  const { data: patterns } = useQuery({
    queryKey: ['spendingPatterns'],
    queryFn: fetchSpendingPatterns,
  });

  const { data: prediction } = useQuery({
    queryKey: ['monthlyPrediction'],
    queryFn: fetchMonthlyPrediction,
  });

  const categoryData = {
    labels: patterns ? Object.keys(patterns.category_distribution) : [],
    datasets: [
      {
        data: patterns ? Object.values(patterns.category_distribution) : [],
        backgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
          '#9966FF',
          '#FF9F40',
        ],
      },
    ],
  };

  const dailyData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [
      {
        label: 'Average Daily Spending',
        data: patterns
          ? Object.values(patterns.daily_avg)
          : Array(7).fill(0),
        backgroundColor: '#36A2EB',
      },
    ],
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>
          Dashboard
        </Typography>
      </Grid>

      {prediction && (
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Next Month's Prediction
            </Typography>
            <Typography variant="body1">
              Predicted Expenses: ${prediction.predicted_amount.toFixed(2)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Range: ${prediction.lower_bound.toFixed(2)} - $
              {prediction.upper_bound.toFixed(2)}
            </Typography>
          </Paper>
        </Grid>
      )}

      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Spending by Category
          </Typography>
          {patterns && <Pie data={categoryData} />}
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Daily Spending Pattern
          </Typography>
          {patterns && <Bar data={dailyData} />}
        </Paper>
      </Grid>

      {patterns?.weekend_vs_weekday && (
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Weekend vs Weekday Spending
            </Typography>
            <Typography>
              Weekend Average: ${patterns.weekend_vs_weekday.weekend_avg.toFixed(2)}
            </Typography>
            <Typography>
              Weekday Average: ${patterns.weekend_vs_weekday.weekday_avg.toFixed(2)}
            </Typography>
          </Paper>
        </Grid>
      )}
    </Grid>
  );
}
