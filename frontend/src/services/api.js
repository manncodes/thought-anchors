const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function fetchProblems(model, solutionType) {
  const res = await fetch(`${API_BASE}/api/problems/${encodeURIComponent(model)}/${encodeURIComponent(solutionType)}`);
  if (!res.ok) throw new Error('Failed to fetch problems');
  return res.json();
}

export async function fetchProblemData(model, solutionType, problemId) {
  const res = await fetch(`${API_BASE}/api/problem/${encodeURIComponent(model)}/${encodeURIComponent(solutionType)}/${encodeURIComponent(problemId)}`);
  if (!res.ok) throw new Error('Failed to fetch problem data');
  return res.json();
}

export async function triggerGeneration(params) {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error('Failed to trigger generation');
  return res.json();
}
