"""Reproduce a misleading score, then exercise the validated implementation."""

from statistics import mean

from ml_metrics import binary_report


def main() -> None:
    actual = [1, 0, 1]
    predicted = [1, 0]
    # Deliberately flawed comparison, isolated here as a review exercise.
    naive = mean(a == p for a, p in zip(actual, predicted))
    print(f"Naive accuracy: {naive:.0%} (only two of three rows compared)")
    try:
        binary_report(actual, predicted)
    except ValueError as error:
        print(f"Validator: {error}")
    print("With the missing prediction supplied:", binary_report(actual, [1, 0, 0]))


if __name__ == "__main__":
    main()
