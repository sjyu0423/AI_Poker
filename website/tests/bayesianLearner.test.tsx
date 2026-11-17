import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { BayesianLearner } from "@/components/opponent/BayesianLearner";

describe("BayesianLearner", () => {
  it("starts from the prior-only preset", () => {
    render(<BayesianLearner />);
    expect(screen.getByText(/Illustrative · Beta–Binomial/i)).toBeInTheDocument();
    expect(screen.getByText("Observed frequency").parentElement).toHaveTextContent("—");
    expect(screen.getByRole("img", { name: /Posterior mean 42\.0%/i })).toBeInTheDocument();
  });

  it("updates after observations and resets", () => {
    render(<BayesianLearner />);
    fireEvent.click(screen.getByRole("button", { name: "Observe Bet" }));
    fireEvent.click(screen.getByRole("button", { name: "Observe Bet" }));
    fireEvent.click(screen.getByRole("button", { name: "Observe Check" }));
    expect(screen.getByText("Observations").parentElement).toHaveTextContent("2 / 3");
    expect(screen.getByText("Observed frequency").parentElement).toHaveTextContent("66.7%");

    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(screen.getByText("Observed frequency").parentElement).toHaveTextContent("—");
    expect(screen.getByText("Observations").parentElement).toHaveTextContent("0 / 0");
  });

  it("loads a sparse preset with stronger data pull", () => {
    render(<BayesianLearner />);
    fireEvent.click(screen.getByRole("button", { name: "3 aggressive bets" }));
    expect(screen.getByText("Observed frequency").parentElement).toHaveTextContent("100.0%");
    expect(screen.getByRole("img", { name: /Posterior mean/i })).toBeInTheDocument();
  });
});
