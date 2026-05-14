"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Prediction = {
  home_team: string;
  away_team: string;
  match_type: string;
  neutral: boolean;
  home_win_probability: number;
  draw_probability: number;
  away_win_probability: number;
  predicted_result: "home_win" | "draw" | "away_win";
  confidence: number;
};

type TeamFeature = {
  team: string;
  team_matches_played: number;
  team_win_rate: number;
  team_draw_rate: number;
  team_loss_rate: number;
  team_avg_goals_scored: number;
  team_avg_goals_conceded: number;
  team_goal_difference_per_match: number;
  team_recent_form_points: number;
  goals_scored: number;
  goals_conceded: number;
  goal_difference: number;
};

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function numberValue(value: number) {
  return Number(value || 0).toFixed(2);
}

function predictionLabel(
  result: Prediction["predicted_result"],
  homeTeam: string,
  awayTeam: string
) {
  if (result === "home_win") return `${homeTeam} win`;
  if (result === "away_win") return `${awayTeam} win`;
  return "Draw";
}

function getTeamFeature(features: TeamFeature[], team: string) {
  return features.find((item) => item.team === team);
}

export default function Home() {
  const [teams, setTeams] = useState<string[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [teamFeatures, setTeamFeatures] = useState<TeamFeature[]>([]);
  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");

  useEffect(() => {
    async function loadData() {
      const [teamsResponse, predictionsResponse, featuresResponse] =
        await Promise.all([
          fetch("/data/app_teams.json"),
          fetch("/data/model_predictions.json"),
          fetch("/data/latest_team_features.json"),
        ]);

      const teamsData: string[] = await teamsResponse.json();
      const predictionsData: Prediction[] = await predictionsResponse.json();
      const featuresData: TeamFeature[] = await featuresResponse.json();

      setTeams(teamsData);
      setPredictions(predictionsData);
      setTeamFeatures(featuresData);

      setHomeTeam(teamsData.includes("Argentina") ? "Argentina" : teamsData[0]);
      setAwayTeam(teamsData.includes("France") ? "France" : teamsData[1]);
    }

    loadData();
  }, []);

  const selectedPrediction = useMemo(() => {
    return predictions.find(
      (prediction) =>
        prediction.home_team === homeTeam && prediction.away_team === awayTeam
    );
  }, [predictions, homeTeam, awayTeam]);

  const homeStats = getTeamFeature(teamFeatures, homeTeam);
  const awayStats = getTeamFeature(teamFeatures, awayTeam);

  const chartData = selectedPrediction
    ? [
        {
          name: `${homeTeam} win`,
          probability: Number(
            (selectedPrediction.home_win_probability * 100).toFixed(1)
          ),
        },
        {
          name: "Draw",
          probability: Number(
            (selectedPrediction.draw_probability * 100).toFixed(1)
          ),
        },
        {
          name: `${awayTeam} win`,
          probability: Number(
            (selectedPrediction.away_win_probability * 100).toFixed(1)
          ),
        },
      ]
    : [];

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <section className="relative overflow-hidden border-b border-white/10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(34,197,94,0.22),_transparent_35%),radial-gradient(circle_at_top_right,_rgba(59,130,246,0.18),_transparent_35%)]" />

        <div className="relative mx-auto max-w-7xl px-6 py-16">
          <div className="mb-8 inline-flex rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-sm text-emerald-200">
            FIFA MatchLab · Machine Learning Football Predictor
          </div>

          <div className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
            <div>
              <h1 className="max-w-4xl text-5xl font-bold tracking-tight text-white md:text-7xl">
                Predict international football match outcomes.
              </h1>

              <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">
                Select two national teams and view win, draw, and loss
                probabilities generated from historical international football
                data.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <span className="rounded-full bg-white/10 px-4 py-2 text-sm text-slate-200">
                  Python
                </span>
                <span className="rounded-full bg-white/10 px-4 py-2 text-sm text-slate-200">
                  Pandas
                </span>
                <span className="rounded-full bg-white/10 px-4 py-2 text-sm text-slate-200">
                  Scikit-learn
                </span>
                <span className="rounded-full bg-white/10 px-4 py-2 text-sm text-slate-200">
                  Next.js
                </span>
                <span className="rounded-full bg-white/10 px-4 py-2 text-sm text-slate-200">
                  Recharts
                </span>
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/[0.06] p-6 shadow-2xl backdrop-blur">
              <p className="text-sm uppercase tracking-[0.25em] text-emerald-300">
                Current prediction
              </p>

              <h2 className="mt-4 text-3xl font-semibold">
                {homeTeam || "Team A"} vs {awayTeam || "Team B"}
              </h2>

              {selectedPrediction && (
                <>
                  <div className="mt-6 rounded-2xl bg-slate-950/80 p-5">
                    <p className="text-sm text-slate-400">Most likely result</p>
                    <p className="mt-2 text-4xl font-bold text-emerald-300">
                      {predictionLabel(
                        selectedPrediction.predicted_result,
                        homeTeam,
                        awayTeam
                      )}
                    </p>
                    <p className="mt-2 text-slate-300">
                      Confidence: {percent(selectedPrediction.confidence)}
                    </p>
                  </div>

                  <div className="mt-5 grid grid-cols-3 gap-3">
                    <div className="rounded-2xl bg-white/10 p-4">
                      <p className="text-xs text-slate-400">{homeTeam}</p>
                      <p className="mt-1 text-2xl font-bold">
                        {percent(selectedPrediction.home_win_probability)}
                      </p>
                    </div>
                    <div className="rounded-2xl bg-white/10 p-4">
                      <p className="text-xs text-slate-400">Draw</p>
                      <p className="mt-1 text-2xl font-bold">
                        {percent(selectedPrediction.draw_probability)}
                      </p>
                    </div>
                    <div className="rounded-2xl bg-white/10 p-4">
                      <p className="text-xs text-slate-400">{awayTeam}</p>
                      <p className="mt-1 text-2xl font-bold">
                        {percent(selectedPrediction.away_win_probability)}
                      </p>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-6 py-10 lg:grid-cols-[0.8fr_1.2fr]">
        <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-6">
          <h2 className="text-2xl font-semibold">Match predictor</h2>
          <p className="mt-2 text-sm leading-6 text-slate-400">
            Choose two teams to generate a neutral-venue World Cup style
            prediction.
          </p>

          <div className="mt-6 space-y-5">
            <label className="block">
              <span className="text-sm text-slate-300">Team A</span>
              <select
                value={homeTeam}
                onChange={(event) => setHomeTeam(event.target.value)}
                className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white outline-none ring-emerald-400/40 focus:ring-4"
              >
                {teams.map((team) => (
                  <option key={team} value={team}>
                    {team}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="text-sm text-slate-300">Team B</span>
              <select
                value={awayTeam}
                onChange={(event) => setAwayTeam(event.target.value)}
                className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white outline-none ring-emerald-400/40 focus:ring-4"
              >
                {teams
                  .filter((team) => team !== homeTeam)
                  .map((team) => (
                    <option key={team} value={team}>
                      {team}
                    </option>
                  ))}
              </select>
            </label>
          </div>

          {selectedPrediction && (
            <div className="mt-6 rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-5">
              <p className="text-sm text-emerald-200">Model explanation</p>
              <p className="mt-2 leading-7 text-slate-200">
                The model compares historical win rate, recent form, attacking
                output, defensive record, match experience, and neutral venue
                context before estimating the most likely result.
              </p>
            </div>
          )}
        </div>

        <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-6">
          <h2 className="text-2xl font-semibold">Prediction probabilities</h2>
          <p className="mt-2 text-sm text-slate-400">
            Probability distribution for the selected matchup.
          </p>

          <div className="mt-6 h-[360px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="name" stroke="#cbd5e1" tick={{ fontSize: 12 }} />
                <YAxis stroke="#cbd5e1" tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    background: "#020617",
                    border: "1px solid rgba(255,255,255,0.15)",
                    borderRadius: "16px",
                    color: "#ffffff",
                  }}
                />
                <Bar
                  dataKey="probability"
                  radius={[12, 12, 0, 0]}
                  fill="#34d399"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-6 pb-16 lg:grid-cols-2">
        <TeamCard team={homeTeam} stats={homeStats} />
        <TeamCard team={awayTeam} stats={awayStats} />
      </section>
    </main>
  );
}

function TeamCard({
  team,
  stats,
}: {
  team: string;
  stats: TeamFeature | undefined;
}) {
  if (!stats) {
    return (
      <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-6">
        <h3 className="text-2xl font-semibold">{team}</h3>
        <p className="mt-4 text-slate-400">Loading team stats...</p>
      </div>
    );
  }

  return (
    <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.25em] text-slate-500">
            Team profile
          </p>
          <h3 className="mt-2 text-3xl font-semibold">{team}</h3>
        </div>

        <div className="rounded-2xl bg-white/10 px-4 py-3 text-right">
          <p className="text-xs text-slate-400">Win rate</p>
          <p className="text-2xl font-bold text-emerald-300">
            {percent(stats.team_win_rate)}
          </p>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-3">
        <Stat label="Matches" value={stats.team_matches_played.toString()} />
        <Stat label="Goals scored" value={stats.goals_scored.toString()} />
        <Stat label="Goals conceded" value={stats.goals_conceded.toString()} />
        <Stat label="Avg goals" value={numberValue(stats.team_avg_goals_scored)} />
        <Stat
          label="Avg conceded"
          value={numberValue(stats.team_avg_goals_conceded)}
        />
        <Stat
          label="Recent form"
          value={numberValue(stats.team_recent_form_points)}
        />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-slate-950/70 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-xl font-semibold text-white">{value}</p>
    </div>
  );
}