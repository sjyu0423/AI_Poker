import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { RegretMatchingDemo } from "@/components/cfr/RegretMatchingDemo";
import { regretMatchingCopy } from "@/content/cfrPage";

describe("RegretMatchingDemo", () => {
  it("renders the formula and defaults", () => {
    render(<RegretMatchingDemo />);
    expect(screen.getByText(/if the sum is positive, else uniform/i)).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(/Positive regret sum/i);
  });

  it("falls back to uniform when regrets are driven non-positive, then resets", () => {
    render(<RegretMatchingDemo />);
    const numberInputs = screen.getAllByRole("spinbutton");
    fireEvent.change(numberInputs[0], { target: { value: "-1" } });
    fireEvent.change(numberInputs[1], { target: { value: "-2" } });
    expect(screen.getByRole("status")).toHaveTextContent(/uniform/i);

    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(numberInputs[0]).toHaveValue(regretMatchingCopy.defaults.regretA);
    expect(numberInputs[1]).toHaveValue(regretMatchingCopy.defaults.regretB);
    expect(screen.getByRole("status")).toHaveTextContent(/Positive regret sum/i);
  });
});
