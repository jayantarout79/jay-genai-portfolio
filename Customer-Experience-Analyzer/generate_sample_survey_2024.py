# generate_sample_survey_2024.py

import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path

NUM_ROWS = 50000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 11, 14)
DAYS_RANGE = (END_DATE - START_DATE).days + 1

CHANNELS = ["Store", "Online", "Support"]
REGIONS = ["US", "EU", "APAC"]
COUNTRIES_BY_REGION = {
    "US": ["US"],
    "EU": ["DE", "FR", "UK", "IT", "ES", "NL"],
    "APAC": ["IN", "JP", "AU", "SG", "KR"],
}
SEGMENTS = ["New Customer", "Existing", "Business"]
PRODUCT_CATEGORIES = ["iPhone", "Mac", "iPad", "Apple Watch", "Services"]

POSITIVE_COMMENTS = [
    "Staff was extremely helpful and friendly.",
    "Checkout was smooth and fast, very satisfied.",
    "Great product experience, everything worked as expected.",
    "Loved the in-store demo, made my decision easy.",
    "Online ordering was simple and delivery was on time.",
    "Support agent resolved my issue quickly and professionally.",
    "Amazing experience overall, I will recommend to others.",
    "Very clear explanations and patient staff.",
    "I was impressed with the product quality and attention to detail.",
    "Great customer service, felt valued as a customer.",
]

NEUTRAL_COMMENTS = [
    "Experience was okay, nothing special.",
    "Got what I needed but it took some time.",
    "The store was a bit crowded but manageable.",
    "Website was fine but could be more intuitive.",
    "Support answered my questions but felt a bit slow.",
    "Overall acceptable experience.",
    "The process was average, neither good nor bad.",
    "Got my issue resolved but had to wait.",
    "Experience was decent but room for improvement.",
    "Things worked but could be smoother.",
]

NEGATIVE_COMMENTS = [
    "Had to wait too long for assistance in store.",
    "Website was confusing and I almost gave up.",
    "Support could not resolve my issue on the first call.",
    "Delivery was delayed and communication was poor.",
    "Felt like no one was available to help in the store.",
    "Checkout process kept failing, very frustrating.",
    "I had to repeat my issue multiple times to different agents.",
    "Product information was unclear, hard to compare options.",
    "The store was overcrowded and disorganized.",
    "Did not feel valued as a customer during this visit.",
]


def generate_comment(score: int) -> str:
    """Return a realistic comment based on NPS score."""
    if score >= 9:
        return random.choice(POSITIVE_COMMENTS)
    elif score >= 7:
        return random.choice(POSITIVE_COMMENTS + NEUTRAL_COMMENTS)
    else:
        return random.choice(NEGATIVE_COMMENTS + NEUTRAL_COMMENTS)


def main():
    rows = []

    for i in range(1, NUM_ROWS + 1):
        # Spread dates across full 2024
        day_offset = random.randint(0, DAYS_RANGE - 1)
        date = START_DATE + timedelta(days=day_offset)

        channel = random.choices(CHANNELS, weights=[0.5, 0.3, 0.2])[0]
        region = random.choices(REGIONS, weights=[0.5, 0.3, 0.2])[0]
        country = random.choice(COUNTRIES_BY_REGION[region])
        segment = random.choices(SEGMENTS, weights=[0.3, 0.5, 0.2])[0]
        product_category = random.choice(PRODUCT_CATEGORIES)

        # NPS distribution: ~55% promoters, 25% passives, 20% detractors
        bucket = random.random()
        if bucket < 0.55:
            nps_score = random.randint(9, 10)
        elif bucket < 0.80:
            nps_score = random.randint(7, 8)
        else:
            nps_score = random.randint(0, 6)

        comment_text = generate_comment(nps_score)
        store_id = f"{region}-{country}-{random.randint(1, 30):03d}"

        rows.append(
            {
                "response_id": f"R{i:05d}",
                "date": date.strftime("%Y-%m-%d"),
                "channel": channel,
                "region": region,
                "country": country,
                "store_id": store_id,
                "nps_score": nps_score,
                "comment_text": comment_text,
                "segment": segment,
                "product_category": product_category,
            }
        )

    df = pd.DataFrame(rows)

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sample_survey.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Generated {len(df)} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()