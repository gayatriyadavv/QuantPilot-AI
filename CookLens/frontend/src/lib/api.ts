import axios from 'axios';
import type { AnalysisResponse } from '@/types';
import { mockAnalyses, getRandomMockAnalysis } from './mockData';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * When true, skip the real API and return mock data.
 * Set NEXT_PUBLIC_USE_MOCK=false in .env.local to use the real API.
 */
const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK !== 'false';

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
  headers: { 'Accept': 'application/json' },
});

/** Simulate network latency for mock responses (ms). */
function mockDelay(min = 2000, max = 3500): Promise<void> {
  const ms = Math.floor(Math.random() * (max - min)) + min;
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Upload a food image for AI analysis.
 */
export async function analyzeImage(file: File): Promise<AnalysisResponse> {
  if (USE_MOCK) {
    await mockDelay();
    return getRandomMockAnalysis();
  }

  try {
    const formData = new FormData();
    formData.append('file', file);

    const { data } = await client.post<AnalysisResponse>('/api/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  } catch (error) {
    console.warn('[CookLens API] analyzeImage failed, falling back to mock:', error);
    return getRandomMockAnalysis();
  }
}

/**
 * Retrieve a previously completed analysis by ID.
 */
export async function getAnalysis(id: string): Promise<AnalysisResponse> {
  if (USE_MOCK) {
    await mockDelay(500, 1000);
    const found = mockAnalyses.find((a) => a.id === id);
    return found ?? getRandomMockAnalysis();
  }

  try {
    const { data } = await client.get<AnalysisResponse>(`/api/analyze/${id}`);
    return data;
  } catch (error) {
    console.warn('[CookLens API] getAnalysis failed, falling back to mock:', error);
    return getRandomMockAnalysis();
  }
}

/**
 * Generate a full recipe from an existing analysis.
 */
export async function generateRecipe(analysisId: string): Promise<AnalysisResponse> {
  if (USE_MOCK) {
    await mockDelay(1500, 2500);
    const found = mockAnalyses.find((a) => a.id === analysisId);
    return found ?? getRandomMockAnalysis();
  }

  try {
    const { data } = await client.post<AnalysisResponse>(`/api/recipe/${analysisId}`);
    return data;
  } catch (error) {
    console.warn('[CookLens API] generateRecipe failed, falling back to mock:', error);
    return getRandomMockAnalysis();
  }
}

/**
 * Fetch analysis history.
 */
export async function getHistory(): Promise<AnalysisResponse[]> {
  if (USE_MOCK) {
    await mockDelay(800, 1200);
    return mockAnalyses;
  }

  try {
    const { data } = await client.get<AnalysisResponse[]>('/api/history');
    return data;
  } catch (error) {
    console.warn('[CookLens API] getHistory failed, falling back to mock:', error);
    return mockAnalyses;
  }
}
