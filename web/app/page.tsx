"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";


type TeamMetadata = Record<
  string,
  {
    team: string;
    display_name: string;
    code: string | null;
    flag: string;
  }
>;

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
  overall_power_score?: number;
};

type HeadToHead = {
  team_a: string;
  team_b: string;
  total_matches: number;
  team_a_wins: number;
  team_b_wins: number;
  draws: number;
  average_goals: number;
  recent_matches: {
    date: string;
    home_team: string;
    away_team: string;
    home_score: number;
    away_score: number;
    tournament: string;
    winner: string;
  }[];
};

type RankingCategory =
  | "overall"
  | "attack"
  | "defence"
  | "recent_form"
  | "win_rate"
  | "goal_difference";

type Rankings = Record<RankingCategory, TeamFeature[]>;

type ModelInsights = {
  accuracy: number;
  train_rows: number;
  test_rows: number;
  classes: string[];
  confusion_matrix: number[][];
  top_features: {
    feature: string;
    importance: number;
  }[];
  classification_report?: Record<
    string,
    | number
    | {
        precision?: number;
        recall?: number;
        "f1-score"?: number;
        support?: number;
      }
  >;
};

type SimulatedMatch = {
  homeTeam: string;
  awayTeam: string;
  winner: string;
  confidence: number;
  homeWinProbability: number;
  drawProbability: number;
  awayWinProbability: number;
  note: string;
};

type SimulatedRound = {
  roundName: string;
  matches: SimulatedMatch[];
};

const rankingCategories: { key: RankingCategory; label: string }[] = [
  { key: "overall", label: "Overall" },
  { key: "attack", label: "Attack" },
  { key: "defence", label: "Defence" },
  { key: "recent_form", label: "Recent form" },
  { key: "win_rate", label: "Win rate" },
  { key: "goal_difference", label: "Goal difference" },
];


let teamMetadataCache: TeamMetadata = {};

function flag(team: string) {
  return teamMetadataCache[team]?.flag || "⚽";
}

function percent(value: number) {
  return `${((value || 0) * 100).toFixed(1)}%`;
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

function resultTeam(
  result: Prediction["predicted_result"],
  homeTeam: string,
  awayTeam: string
) {
  if (result === "home_win") return homeTeam;
  if (result === "away_win") return awayTeam;
  return "Draw";
}

function getTeamFeature(features: TeamFeature[], team: string) {
  return features.find((item) => item.team === team);
}

function getPrediction(
  predictions: Prediction[],
  homeTeam: string,
  awayTeam: string
) {
  return predictions.find(
    (prediction) =>
      prediction.home_team === homeTeam && prediction.away_team === awayTeam
  );
}

function powerScore(stats: TeamFeature | undefined) {
  if (!stats) return 0;

  return (
    stats.team_win_rate * 45 +
    stats.team_recent_form_points * 15 +
    stats.team_goal_difference_per_match * 20 +
    stats.team_avg_goals_scored * 10 -
    stats.team_avg_goals_conceded * 10
  );
}

function buildReasons(
  homeTeam: string,
  awayTeam: string,
  homeStats: TeamFeature | undefined,
  awayStats: TeamFeature | undefined,
  prediction: Prediction | undefined
) {
  if (!homeStats || !awayStats || !prediction) {
    return ["The model is loading the team data for this matchup."];
  }

  const winner = resultTeam(prediction.predicted_result, homeTeam, awayTeam);

  if (winner === "Draw") {
    return [
      "The model sees this as a balanced matchup with no clear winner.",
      `The probability gap between ${homeTeam} and ${awayTeam} is relatively small.`,
      "Draws are common in football when attacking and defensive indicators are closely matched.",
    ];
  }

  const winnerStats = winner === homeTeam ? homeStats : awayStats;
  const loserStats = winner === homeTeam ? awayStats : homeStats;
  const loser = winner === homeTeam ? awayTeam : homeTeam;

  const reasons: string[] = [];

  if (winnerStats.team_recent_form_points > loserStats.team_recent_form_points) {
    reasons.push(
      `${winner} has stronger recent form than ${loser}, based on average points from recent matches.`
    );
  }

  if (winnerStats.team_win_rate > loserStats.team_win_rate) {
    reasons.push(
      `${winner} has a stronger historical win rate across the dataset.`
    );
  }

  if (
    winnerStats.team_goal_difference_per_match >
    loserStats.team_goal_difference_per_match
  ) {
    reasons.push(
      `${winner} has a better goal difference per match, suggesting stronger overall balance.`
    );
  }

  if (winnerStats.team_avg_goals_scored > loserStats.team_avg_goals_scored) {
    reasons.push(
      `${winner} has produced more goals per match on average.`
    );
  }

  if (
    winnerStats.team_avg_goals_conceded <
    loserStats.team_avg_goals_conceded
  ) {
    reasons.push(
      `${winner} has a stronger defensive record, conceding fewer goals per match.`
    );
  }

  if (reasons.length === 0) {
    reasons.push(
      `${winner} is predicted to edge this matchup based on the combined model features.`
    );
  }

  return reasons.slice(0, 4);
}

function upsetAlert(
  homeTeam: string,
  awayTeam: string,
  homeStats: TeamFeature | undefined,
  awayStats: TeamFeature | undefined,
  prediction: Prediction | undefined
) {
  if (!homeStats || !awayStats || !prediction) {
    return {
      level: "Loading",
      text: "Waiting for model output.",
    };
  }

  const winner = resultTeam(prediction.predicted_result, homeTeam, awayTeam);
  const homePower = powerScore(homeStats);
  const awayPower = powerScore(awayStats);

  if (winner === "Draw") {
    return {
      level: "Volatile",
      text: "The model sees this as a close matchup, which increases uncertainty.",
    };
  }

  const winnerPower = winner === homeTeam ? homePower : awayPower;
  const opponentPower = winner === homeTeam ? awayPower : homePower;
  const probabilityGap = Math.abs(
    prediction.home_win_probability - prediction.away_win_probability
  );

  if (winnerPower < opponentPower - 5) {
    return {
      level: "Upset candidate",
      text: `${winner} is predicted to win despite having a lower overall power score. This is a strong upset signal.`,
    };
  }

  if (prediction.confidence < 0.45 || probabilityGap < 0.08) {
    return {
      level: "High uncertainty",
      text: "The model is not strongly confident. Small changes in form or team strength could flip the outcome.",
    };
  }

  return {
    level: "Stable pick",
    text: "The predicted result is reasonably aligned with the broader team indicators.",
  };
}

function roundName(teamCount: number) {
  if (teamCount === 16) return "Round of 16";
  if (teamCount === 8) return "Quarter-finals";
  if (teamCount === 4) return "Semi-finals";
  if (teamCount === 2) return "Final";
  return "Knockout round";
}

function shuffleArray(items: string[]) {
  return [...items].sort(() => Math.random() - 0.5);
}

export default function Home() {
  const [teams, setTeams] = useState<string[]>([]);
  const [, setTeamMetadata] = useState<TeamMetadata>({});
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [teamFeatures, setTeamFeatures] = useState<TeamFeature[]>([]);
  const [headToHead, setHeadToHead] = useState<Record<string, HeadToHead>>({});
  const [rankings, setRankings] = useState<Rankings | null>(null);
  const [modelInsights, setModelInsights] = useState<ModelInsights | null>(
    
    null
  );

  
  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");
  const [activeRanking, setActiveRanking] =
    useState<RankingCategory>("overall");

  const [simTeams, setSimTeams] = useState<string[]>([]);
  const [simTeamToAdd, setSimTeamToAdd] = useState("");
  const [bracketRounds, setBracketRounds] = useState<SimulatedRound[]>([]);

  useEffect(() => {
    async function loadData() {
      const [
        teamsResponse,
        predictionsResponse,
        featuresResponse,
        headToHeadResponse,
        rankingsResponse,
        modelInsightsResponse,
        teamMetadataResponse,
      ] = await Promise.all([
        fetch("/data/team_metadata.json"),
        fetch("/data/app_teams.json"),
        fetch("/data/model_predictions.json"),
        fetch("/data/latest_team_features.json"),
        fetch("/data/head_to_head.json"),
        fetch("/data/rankings.json"),
        fetch("/data/model_insights.json"),
      ]);

      const rawTeamsData = await teamsResponse.json();
      const predictionsData: Prediction[] = await predictionsResponse.json();
      const featuresData: TeamFeature[] = await featuresResponse.json();
      const headToHeadData: Record<string, HeadToHead> =
        await headToHeadResponse.json();
      const rawRankings = await rankingsResponse.json();

const rankingsData: Rankings = {
  overall: Array.isArray(rawRankings?.overall) ? rawRankings.overall : [],
  attack: Array.isArray(rawRankings?.attack) ? rawRankings.attack : [],
  defence: Array.isArray(rawRankings?.defence) ? rawRankings.defence : [],
  recent_form: Array.isArray(rawRankings?.recent_form)
    ? rawRankings.recent_form
    : [],
  win_rate: Array.isArray(rawRankings?.win_rate) ? rawRankings.win_rate : [],
  goal_difference: Array.isArray(rawRankings?.goal_difference)
    ? rawRankings.goal_difference
    : [],
};
      const rawModelInsights = await modelInsightsResponse.json();
      const teamMetadataData: TeamMetadata = await teamMetadataResponse.json();

    const teamsData: string[] = Array.isArray(rawTeamsData)
      ? rawTeamsData
      : Object.values(rawTeamsData)
          .map((item: any) => {
            if (typeof item === "string") return item;
            return item?.team;
          })
          .filter((team): team is string => Boolean(team))
          .sort();

      const modelInsightsData: ModelInsights = {
        accuracy: Number(rawModelInsights?.accuracy ?? 0),
        train_rows: Number(rawModelInsights?.train_rows ?? 0),
        test_rows: Number(rawModelInsights?.test_rows ?? 0),
        classes: Array.isArray(rawModelInsights?.classes)
          ? rawModelInsights.classes
          : [],
        confusion_matrix: Array.isArray(rawModelInsights?.confusion_matrix)
          ? rawModelInsights.confusion_matrix
          : [],
        top_features: Array.isArray(rawModelInsights?.top_features)
          ? rawModelInsights.top_features
          : [],
        classification_report: rawModelInsights?.classification_report ?? {},
      };


      setTeams(teamsData);
      setPredictions(predictionsData);
      setTeamFeatures(featuresData);
      setHeadToHead(headToHeadData);
      setRankings(rankingsData);
      setModelInsights(modelInsightsData);

      teamMetadataCache = teamMetadataData;
      setTeamMetadata(teamMetadataData);

      const defaultHome = teamsData.includes("Argentina")
        ? "Argentina"
        : teamsData[0] ?? "";

      const defaultAway =
      teamsData.includes("France") && defaultHome !== "France"
        ? "France"
        : teamsData.find((team) => team !== defaultHome) ?? "";

      setHomeTeam(defaultHome);
      setAwayTeam(defaultAway);

      const defaultSimTeams = (
  rankingsData.overall.length ? rankingsData.overall : featuresData
)
  .slice(0, 8)
  .map((item) => item.team)
  .filter(Boolean);

      setSimTeams(defaultSimTeams);
      setSimTeamToAdd(teamsData[0] || "");
    }

    loadData();
  }, []);

  const selectedPrediction = useMemo(() => {
    return getPrediction(predictions, homeTeam, awayTeam);
  }, [predictions, homeTeam, awayTeam]);

  const homeStats = getTeamFeature(teamFeatures, homeTeam);
  const awayStats = getTeamFeature(teamFeatures, awayTeam);

  const selectedH2H = headToHead[`${homeTeam}__${awayTeam}`];

  const reasons = buildReasons(
    homeTeam,
    awayTeam,
    homeStats,
    awayStats,
    selectedPrediction
  );

  const alert = upsetAlert(
    homeTeam,
    awayTeam,
    homeStats,
    awayStats,
    selectedPrediction
  );

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

  const radarData =
    homeStats && awayStats
      ? [
          {
            metric: "Win rate",
            [homeTeam]: Number((homeStats.team_win_rate * 100).toFixed(1)),
            [awayTeam]: Number((awayStats.team_win_rate * 100).toFixed(1)),
          },
          {
            metric: "Recent form",
            [homeTeam]: Number(
              ((homeStats.team_recent_form_points / 3) * 100).toFixed(1)
            ),
            [awayTeam]: Number(
              ((awayStats.team_recent_form_points / 3) * 100).toFixed(1)
            ),
          },
          {
            metric: "Attack",
            [homeTeam]: Number(
              (Math.min(homeStats.team_avg_goals_scored, 4) * 25).toFixed(1)
            ),
            [awayTeam]: Number(
              (Math.min(awayStats.team_avg_goals_scored, 4) * 25).toFixed(1)
            ),
          },
          {
            metric: "Defence",
            [homeTeam]: Number(
              (Math.max(0, 4 - homeStats.team_avg_goals_conceded) * 25).toFixed(
                1
              )
            ),
            [awayTeam]: Number(
              (Math.max(0, 4 - awayStats.team_avg_goals_conceded) * 25).toFixed(
                1
              )
            ),
          },
          {
            metric: "Goal diff",
            [homeTeam]: Number(
              (Math.max(0, homeStats.team_goal_difference_per_match + 2) * 25).toFixed(1)
            ),
            [awayTeam]: Number(
              (Math.max(0, awayStats.team_goal_difference_per_match + 2) * 25).toFixed(1)
            ),
          },
        ]
      : [];

  const rankingRows = rankings ? rankings[activeRanking] || [] : [];

  function simulateMatch(home: string, away: string): SimulatedMatch {
    const prediction = getPrediction(predictions, home, away);
    const homeTeamStats = getTeamFeature(teamFeatures, home);
    const awayTeamStats = getTeamFeature(teamFeatures, away);

    if (!prediction) {
      const winner =
        powerScore(homeTeamStats) >= powerScore(awayTeamStats) ? home : away;

      return {
        homeTeam: home,
        awayTeam: away,
        winner,
        confidence: 0.5,
        homeWinProbability: 0.5,
        drawProbability: 0,
        awayWinProbability: 0.5,
        note: "Fallback power-score tie-breaker used.",
      };
    }

    let winner = resultTeam(prediction.predicted_result, home, away);
    let note = "Winner selected from model prediction.";

    if (winner === "Draw") {
      winner =
        powerScore(homeTeamStats) >= powerScore(awayTeamStats) ? home : away;
      note = "Draw predicted, so knockout tie-breaker used team power score.";
    }

    return {
      homeTeam: home,
      awayTeam: away,
      winner,
      confidence: prediction.confidence,
      homeWinProbability: prediction.home_win_probability,
      drawProbability: prediction.draw_probability,
      awayWinProbability: prediction.away_win_probability,
      note,
    };
  }

  function runSimulation() {
    let size = 0;

    if (simTeams.length >= 16) size = 16;
    else if (simTeams.length >= 8) size = 8;
    else if (simTeams.length >= 4) size = 4;
    else if (simTeams.length >= 2) size = 2;

    if (!size) return;

    let currentTeams = simTeams.slice(0, size);
    const rounds: SimulatedRound[] = [];

    while (currentTeams.length > 1) {
      const matches: SimulatedMatch[] = [];
      const winners: string[] = [];

      for (let index = 0; index < currentTeams.length; index += 2) {
        const match = simulateMatch(currentTeams[index], currentTeams[index + 1]);
        matches.push(match);
        winners.push(match.winner);
      }

      rounds.push({
        roundName: roundName(currentTeams.length),
        matches,
      });

      currentTeams = winners;
    }

    setBracketRounds(rounds);
  }

  function loadTopTeams(count: number) {
  const source = rankings?.overall?.length ? rankings.overall : teamFeatures;

  const selected = source
    .slice(0, count)
    .map((item) => item.team)
    .filter(Boolean);

  setSimTeams(selected);
  setBracketRounds([]);
}

  function loadRandomTeams(count: number) {
    const selected = shuffleArray(teams).slice(0, count);
    setSimTeams(selected);
    setBracketRounds([]);
  }

  function addSimTeam() {
    if (!simTeamToAdd) return;
    if (simTeams.includes(simTeamToAdd)) return;
    if (simTeams.length >= 16) return;

    setSimTeams((current) => [...current, simTeamToAdd]);
    setBracketRounds([]);
  }

  function removeSimTeam(team: string) {
    setSimTeams((current) => current.filter((item) => item !== team));
    setBracketRounds([]);
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <section className="relative overflow-hidden border-b border-white/10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(16,185,129,0.24),_transparent_35%),radial-gradient(circle_at_top_right,_rgba(59,130,246,0.2),_transparent_35%),linear-gradient(135deg,_#020617,_#07111f_45%,_#020617)]" />
        <div className="absolute inset-0 opacity-[0.08] [background-image:linear-gradient(90deg,_#fff_1px,_transparent_1px),linear-gradient(#fff_1px,_transparent_1px)] [background-size:80px_80px]" />

        <div className="relative mx-auto max-w-7xl px-6 py-8">
          <nav className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="grid h-11 w-11 place-items-center rounded-2xl bg-emerald-400 text-2xl shadow-lg shadow-emerald-500/30">
                ⚽
              </div>
              <div>
                <p className="text-lg font-bold">FIFA MatchLab</p>
                <p className="text-xs text-slate-400">
                  Football analytics and prediction platform
                </p>
              </div>
            </div>

            <div className="flex flex-wrap gap-2 text-sm text-slate-300">
              <a className="rounded-full bg-white/10 px-4 py-2 hover:bg-white/15" href="#predictor">
                Predictor
              </a>
              <a className="rounded-full bg-white/10 px-4 py-2 hover:bg-white/15" href="#simulator">
                Simulator
              </a>
              <a className="rounded-full bg-white/10 px-4 py-2 hover:bg-white/15" href="#rankings">
                Rankings
              </a>
              <a className="rounded-full bg-white/10 px-4 py-2 hover:bg-white/15" href="#model">
                Model
              </a>
            </div>
          </nav>

          <div className="grid gap-10 py-16 lg:grid-cols-[1.15fr_0.85fr] lg:items-center">
            <div>
              <div className="mb-6 inline-flex rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-sm text-emerald-200">
                Machine learning football predictor
              </div>

              <h1 className="max-w-4xl text-5xl font-black tracking-tight md:text-7xl">
                Predict matches. Compare teams. Simulate tournaments.
              </h1>

              <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">
                FIFA MatchLab uses historical international football data to
                estimate match outcomes, explain model predictions, compare
                national teams, and run knockout-style tournament simulations.
              </p>

              <div className="mt-8 grid gap-3 sm:grid-cols-3">
                <HeroStat label="Teams" value={teams.length ? `${teams.length}+` : "..."} />
                <HeroStat
                  label="Model accuracy"
                  value={modelInsights ? percent(modelInsights.accuracy) : "..."}
                />
                <HeroStat
                  label="Test matches"
                  value={modelInsights ? (modelInsights.test_rows ?? 0).toString() : "..."}
                />
              </div>
            </div>

            <div className="rounded-[2rem] border border-white/10 bg-white/[0.06] p-6 shadow-2xl shadow-emerald-950/40 backdrop-blur">
              <p className="text-sm uppercase tracking-[0.25em] text-emerald-300">
                Featured matchup
              </p>

              <h2 className="mt-4 text-3xl font-bold">
                {flag(homeTeam)} {homeTeam || "Team A"} vs {flag(awayTeam)}{" "}
                {awayTeam || "Team B"}
              </h2>

              {selectedPrediction && (
                <>
                  <div className="mt-6 rounded-3xl bg-slate-950/80 p-5">
                    <p className="text-sm text-slate-400">Most likely result</p>
                    <p className="mt-2 text-4xl font-black text-emerald-300">
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
                    <MiniProbability
                      label={homeTeam}
                      value={selectedPrediction.home_win_probability}
                    />
                    <MiniProbability
                      label="Draw"
                      value={selectedPrediction.draw_probability}
                    />
                    <MiniProbability
                      label={awayTeam}
                      value={selectedPrediction.away_win_probability}
                    />
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </section>

      <section
        id="predictor"
        className="mx-auto grid max-w-7xl gap-6 px-6 py-10 lg:grid-cols-[0.8fr_1.2fr]"
      >
        <div className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
          <SectionHeader
            eyebrow="Match predictor"
            title="Choose a matchup"
            text="Select two national teams and view the model's neutral-venue World Cup style prediction."
          />

          <div className="mt-6 space-y-5">
            <TeamSelect
              label="Team A"
              value={homeTeam}
              teams={teams}
              onChange={setHomeTeam}
            />

            <TeamSelect
              label="Team B"
              value={awayTeam}
              teams={teams.filter((team) => team !== homeTeam)}
              onChange={setAwayTeam}
            />
          </div>

          <div className="mt-6 rounded-3xl border border-amber-300/20 bg-amber-300/10 p-5">
            <p className="text-sm font-semibold text-amber-200">
              {alert.level}
            </p>
            <p className="mt-2 leading-7 text-slate-200">{alert.text}</p>
          </div>
        </div>

        <div className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
          <SectionHeader
            eyebrow="Probability distribution"
            title="Prediction probabilities"
            text="The model outputs win, draw, and loss probabilities for the selected fixture."
          />

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
                <Bar dataKey="probability" radius={[14, 14, 0, 0]} fill="#34d399" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-6 pb-10 lg:grid-cols-[1fr_1fr]">
        <div className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
          <SectionHeader
            eyebrow="Explainability"
            title="Why this prediction?"
            text="A readable explanation based on form, attack, defence, goal difference, and win rate."
          />

          <div className="mt-6 space-y-3">
            {reasons.map((reason) => (
              <div
                key={reason}
                className="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-4 text-slate-200"
              >
                {reason}
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
          <SectionHeader
            eyebrow="Comparison radar"
            title={`${homeTeam || "Team A"} vs ${awayTeam || "Team B"}`}
            text="A normalized view of each team's key performance indicators."
          />

          <div className="mt-6 h-[330px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.16)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "#cbd5e1", fontSize: 12 }} />
                <Radar
                  name={homeTeam}
                  dataKey={homeTeam}
                  stroke="#34d399"
                  fill="#34d399"
                  fillOpacity={0.25}
                />
                <Radar
                  name={awayTeam}
                  dataKey={awayTeam}
                  stroke="#60a5fa"
                  fill="#60a5fa"
                  fillOpacity={0.18}
                />
                <Tooltip
                  contentStyle={{
                    background: "#020617",
                    border: "1px solid rgba(255,255,255,0.15)",
                    borderRadius: "16px",
                    color: "#ffffff",
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-6 pb-10 lg:grid-cols-2">
        <TeamCard team={homeTeam} stats={homeStats} />
        <TeamCard team={awayTeam} stats={awayStats} />
      </section>

      <section className="mx-auto max-w-7xl px-6 pb-10">
        <div className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
          <SectionHeader
            eyebrow="Head-to-head"
            title={`${flag(homeTeam)} ${homeTeam || "Team A"} vs ${flag(awayTeam)} ${awayTeam || "Team B"}`}
            text="Historical meetings between the selected teams from the dataset."
          />

          {selectedH2H ? (
            <>
              <div className="mt-6 grid gap-4 sm:grid-cols-4">
                <Stat label="Matches" value={selectedH2H.total_matches.toString()} />
                <Stat label={`${homeTeam} wins`} value={selectedH2H.team_a_wins.toString()} />
                <Stat label="Draws" value={selectedH2H.draws.toString()} />
                <Stat label={`${awayTeam} wins`} value={selectedH2H.team_b_wins.toString()} />
              </div>

              <div className="mt-6 grid gap-3">
                {selectedH2H.recent_matches.length ? (
                  selectedH2H.recent_matches.map((match) => (
                    <div
                      key={`${match.date}-${match.home_team}-${match.away_team}`}
                      className="grid gap-3 rounded-2xl bg-slate-950/70 p-4 text-sm md:grid-cols-[1fr_1.5fr_1fr]"
                    >
                      <p className="text-slate-400">{match.date}</p>
                      <p className="font-semibold">
                        {flag(match.home_team)} {match.home_team} {match.home_score} -{" "}
                        {match.away_score} {flag(match.away_team)} {match.away_team}
                      </p>
                      <p className="text-slate-400">{match.tournament}</p>
                    </div>
                  ))
                ) : (
                  <p className="rounded-2xl bg-slate-950/70 p-4 text-slate-400">
                    No previous head-to-head matches found for this pair.
                  </p>
                )}
              </div>
            </>
          ) : (
            <p className="mt-6 rounded-2xl bg-slate-950/70 p-4 text-slate-400">
              Loading head-to-head data...
            </p>
          )}
        </div>
      </section>

      <section id="simulator" className="mx-auto max-w-7xl px-6 pb-10">
        <div className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
          <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
            <div>
              <SectionHeader
                eyebrow="Knockout simulator"
                title="Simulate a tournament"
                text="Choose 2, 4, 8, or 16 teams and let the model simulate a knockout bracket."
              />

              <div className="mt-6 flex flex-wrap gap-3">
                <button onClick={() => loadTopTeams(8)} className="rounded-full bg-white/10 px-4 py-2 text-sm hover:bg-white/15">
                  Load top 8
                </button>
                <button onClick={() => loadTopTeams(16)} className="rounded-full bg-white/10 px-4 py-2 text-sm hover:bg-white/15">
                  Load top 16
                </button>
                <button onClick={() => loadRandomTeams(8)} className="rounded-full bg-white/10 px-4 py-2 text-sm hover:bg-white/15">
                  Random 8
                </button>
              </div>

              <div className="mt-5 flex gap-3">
                <select
                  value={simTeamToAdd}
                  onChange={(event) => setSimTeamToAdd(event.target.value)}
                  className="min-w-0 flex-1 rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white outline-none"
                >
                  {teams
                    .filter((team) => !simTeams.includes(team))
                    .map((team) => (
                      <option key={team} value={team}>
                        {team}
                      </option>
                    ))}
                </select>

                <button
                  onClick={addSimTeam}
                  className="rounded-2xl bg-emerald-400 px-5 py-3 font-semibold text-slate-950 hover:bg-emerald-300"
                >
                  Add
                </button>
              </div>

              <div className="mt-5 flex flex-wrap gap-2">
                {simTeams.map((team) => (
                  <button
                    key={team}
                    onClick={() => removeSimTeam(team)}
                    className="rounded-full border border-white/10 bg-slate-950/70 px-3 py-2 text-sm text-slate-200 hover:bg-red-500/20"
                  >
                    {flag(team)} {team} ×
                  </button>
                ))}
              </div>

              <button
                onClick={runSimulation}
                className="mt-6 w-full rounded-2xl bg-emerald-400 px-5 py-4 font-bold text-slate-950 shadow-lg shadow-emerald-500/20 hover:bg-emerald-300"
              >
                Run knockout simulation
              </button>
            </div>

            <div>
              {bracketRounds.length ? (
                <div className="space-y-5">
                  {bracketRounds.map((round) => (
                    <div key={round.roundName} className="rounded-3xl bg-slate-950/70 p-5">
                      <h3 className="text-xl font-bold">{round.roundName}</h3>
                      <div className="mt-4 grid gap-3">
                        {round.matches.map((match) => (
                          <div
                            key={`${round.roundName}-${match.homeTeam}-${match.awayTeam}`}
                            className="rounded-2xl border border-white/10 bg-white/[0.04] p-4"
                          >
                            <div className="flex flex-wrap items-center justify-between gap-3">
                              <p className="font-semibold">
                                {flag(match.homeTeam)} {match.homeTeam} vs{" "}
                                {flag(match.awayTeam)} {match.awayTeam}
                              </p>
                              <p className="rounded-full bg-emerald-400/15 px-3 py-1 text-sm text-emerald-200">
                                Winner: {flag(match.winner)} {match.winner}
                              </p>
                            </div>
                            <p className="mt-2 text-sm text-slate-400">
                              Confidence: {percent(match.confidence)} · {match.note}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="grid min-h-[420px] place-items-center rounded-3xl border border-dashed border-white/15 bg-slate-950/50 p-8 text-center">
                  <div>
                    <p className="text-5xl">🏆</p>
                    <h3 className="mt-4 text-2xl font-bold">
                      Your simulated bracket will appear here
                    </h3>
                    <p className="mt-2 text-slate-400">
                      Load teams, run the simulation, and the model will predict
                      each knockout round.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <section id="rankings" className="mx-auto max-w-7xl px-6 pb-10">
        <div className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
          <SectionHeader
            eyebrow="Team rankings"
            title="Explore model-based team profiles"
            text="Rank teams by attack, defence, recent form, win rate, and overall power score."
          />

          <div className="mt-6 flex flex-wrap gap-2">
            {rankingCategories.map((category) => (
              <button
                key={category.key}
                onClick={() => setActiveRanking(category.key)}
                className={`rounded-full px-4 py-2 text-sm ${
                  activeRanking === category.key
                    ? "bg-emerald-400 text-slate-950"
                    : "bg-white/10 text-slate-200 hover:bg-white/15"
                }`}
              >
                {category.label}
              </button>
            ))}
          </div>

          <div className="mt-6 grid gap-3">
            {rankingRows.slice(0, 10).map((team, index) => (
              <div
                key={`${activeRanking}-${team.team}`}
                className="grid items-center gap-4 rounded-2xl bg-slate-950/70 p-4 md:grid-cols-[70px_1fr_120px_120px_120px]"
              >
                <p className="text-2xl font-black text-slate-500">
                  #{index + 1}
                </p>
                <p className="text-lg font-bold">
                  {flag(team.team)} {team.team}
                </p>
                <p className="text-sm text-slate-300">
                  Win: {percent(team.team_win_rate)}
                </p>
                <p className="text-sm text-slate-300">
                  Attack: {numberValue(team.team_avg_goals_scored)}
                </p>
                <p className="text-sm text-slate-300">
                  Defence: {numberValue(team.team_avg_goals_conceded)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="model" className="mx-auto max-w-7xl px-6 pb-16">
        <div className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
          <SectionHeader
            eyebrow="Model performance"
            title="How the prediction model performs"
            text="A transparent look at model accuracy, training data, test data, and influential features."
          />

          <div className="mt-6 grid gap-4 md:grid-cols-3">
            <Stat
              label="Accuracy"
              value={modelInsights ? percent(modelInsights.accuracy) : "..."}
            />
            <Stat
              label="Training rows"
              value={modelInsights ? (modelInsights.train_rows ?? 0).toLocaleString() : "..."}
            />
            <Stat
              label="Testing rows"
              value={modelInsights ? (modelInsights.test_rows ?? 0).toLocaleString() : "..."}
            />
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <div className="rounded-3xl bg-slate-950/70 p-5">
              <h3 className="text-xl font-bold">Most important numeric features</h3>
              <div className="mt-4 space-y-3">
                {modelInsights?.top_features?.slice(0, 8).map((feature) => (
                  <div key={feature.feature}>
                    <div className="mb-1 flex justify-between text-sm">
                      <span className="text-slate-300">{feature.feature}</span>
                      <span className="text-emerald-300">{feature.importance}</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full bg-emerald-400"
                        style={{
                          width: `${Math.min(feature.importance * 100, 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-3xl bg-slate-950/70 p-5">
              <h3 className="text-xl font-bold">Classification report</h3>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[480px] text-left text-sm">
                  <thead className="text-slate-400">
                    <tr>
                      <th className="py-2">Class</th>
                      <th className="py-2">Precision</th>
                      <th className="py-2">Recall</th>
                      <th className="py-2">F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {["home_win", "draw", "away_win"].map((label) => {
                      const report = modelInsights?.classification_report?.[label];

                      if (!report || typeof report === "number") return null;

                      return (
                        <tr key={label} className="border-t border-white/10">
                          <td className="py-3 font-semibold">{label}</td>
                          <td className="py-3">{numberValue(report.precision || 0)}</td>
                          <td className="py-3">{numberValue(report.recall || 0)}</td>
                          <td className="py-3">{numberValue(report["f1-score"] || 0)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <p className="mt-4 text-sm leading-6 text-slate-400">
                Football is naturally difficult to predict because team selection,
                injuries, tactics, and match context can change quickly. This model
                is intended as a transparent analytics project rather than a betting
                tool.
              </p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

function SectionHeader({
  eyebrow,
  title,
  text,
}: {
  eyebrow: string;
  title: string;
  text: string;
}) {
  return (
    <div>
      <p className="text-sm uppercase tracking-[0.25em] text-emerald-300">
        {eyebrow}
      </p>
      <h2 className="mt-2 text-3xl font-black">{title}</h2>
      <p className="mt-2 max-w-2xl leading-7 text-slate-400">{text}</p>
    </div>
  );
}

function HeroStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/[0.06] p-5">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-2 text-3xl font-black text-white">{value}</p>
    </div>
  );
}

function MiniProbability({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-2xl bg-white/10 p-4">
      <p className="truncate text-xs text-slate-400">{label}</p>
      <p className="mt-1 text-2xl font-black">{percent(value)}</p>
    </div>
  );
}

function TeamSelect({
  label,
  value,
  teams,
  onChange,
}: {
  label: string;
  value: string;
  teams: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-sm text-slate-300">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-900 px-4 py-3 text-white outline-none ring-emerald-400/40 focus:ring-4"
      >
        {teams.map((team) => (
          <option key={team} value={team}>
            {flag(team)} {team}
          </option>
        ))}
      </select>
    </label>
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
      <div className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
        <h3 className="text-2xl font-semibold">{team}</h3>
        <p className="mt-4 text-slate-400">Loading team stats...</p>
      </div>
    );
  }

  return (
    <div className="rounded-[2rem] border border-white/10 bg-white/[0.04] p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.25em] text-slate-500">
            Team profile
          </p>
          <h3 className="mt-2 text-3xl font-black">
            {flag(team)} {team}
          </h3>
        </div>

        <div className="rounded-2xl bg-white/10 px-4 py-3 text-right">
          <p className="text-xs text-slate-400">Win rate</p>
          <p className="text-2xl font-black text-emerald-300">
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
      <p className="mt-1 text-xl font-bold text-white">{value}</p>
    </div>
  );
}