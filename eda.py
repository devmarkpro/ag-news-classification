import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import multiprocessing as mp
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
additional_stopwords = {
    "said",
    "says",
    "say",
    "told",
    "according",
    "reuters",
    "ap",
    "afp",
    "news",
    "report",
    "reports",
    "reported",
    "reporting",
    "sources",
    "source",
    "officials",
    "official",
    "spokesman",
    "spokesperson",
    "statement",
    "announced",
    "announce",
    "announcement",
    "conference",
    "press",
    "media",
    "journalist",
    "journalists",
    "article",
    "story",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "today",
    "yesterday",
    "tomorrow",
    "week",
    "month",
    "year",
    "years",
    "time",
    "times",
    "day",
    "days",
    "hour",
    "hours",
    "minute",
    "minutes",
    "people",
    "person",
    "man",
    "woman",
    "men",
    "women",
    "group",
    "groups",
    "number",
    "numbers",
    "percent",
    "percentage",
    "million",
    "billion",
    "thousand",
    "hundreds",
    "dozens",
    "several",
    "many",
    "much",
    "most",
    "some",
    "few",
    "little",
    "large",
    "small",
    "big",
    "major",
    "minor",
    "new",
    "old",
    "first",
    "last",
    "next",
    "previous",
    "recent",
    "latest",
    "early",
    "late",
    "long",
    "short",
    "high",
    "low",
    "good",
    "bad",
    "best",
    "worst",
    "better",
    "worse",
    "great",
    "small",
    "large",
}
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


class AGNewsEDA:
    def __init__(self, data_path="data/"):
        self.data_path = data_path
        self.class_names = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
        self.colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

        os.makedirs("outputs/eda", exist_ok=True)

    def load_data(self):
        print("Loading data...")

        self.train_df = pd.read_csv(f"{self.data_path}/train.csv")
        self.test_df = pd.read_csv(f"{self.data_path}/test.csv")

        self.full_df = pd.concat([self.train_df, self.test_df], ignore_index=True)

        self.full_df["text"] = (
            self.full_df["Title"].astype(str)
            + ". "
            + self.full_df["Description"].astype(str)
        )
        self.full_df["class_name"] = self.full_df["Class Index"].map(self.class_names)

        self.full_df["title_length"] = self.full_df["Title"].str.len()
        self.full_df["desc_length"] = self.full_df["Description"].str.len()
        self.full_df["text_length"] = self.full_df["text"].str.len()
        self.full_df["word_count"] = self.full_df["text"].str.split().str.len()

        print(f"Loaded {len(self.train_df)} training samples")
        print(f"Loaded {len(self.test_df)} test samples")
        print(f"Total samples: {len(self.full_df)}")

    def create_overview_dashboard(self):
        print("\nCreating overview dashboard...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "AG News Dataset - Comprehensive Overview",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        # 1. Class Distribution
        class_counts = self.full_df["class_name"].value_counts()
        axes[0, 0].pie(
            class_counts.values,
            labels=class_counts.index,
            autopct="%1.1f%%",
            colors=self.colors,
            startangle=90,
        )
        axes[0, 0].set_title("Class Distribution", fontsize=14, fontweight="bold")

        # 2. Text Length Distribution
        axes[0, 1].hist(
            self.full_df["text_length"],
            bins=50,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        axes[0, 1].set_title("Text Length Distribution", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Text Length (characters)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].axvline(
            self.full_df["text_length"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {self.full_df["text_length"].mean():.0f}',
        )
        axes[0, 1].legend()

        # 3. Word Count by Class
        box_data = [
            self.full_df[self.full_df["class_name"] == class_name]["word_count"]
            for class_name in self.class_names.values()
        ]
        bp = axes[0, 2].boxplot(
            box_data, labels=list(self.class_names.values()), patch_artist=True
        )
        for patch, color in zip(bp["boxes"], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0, 2].set_title(
            "Word Count Distribution by Class", fontsize=14, fontweight="bold"
        )
        axes[0, 2].set_ylabel("Word Count")
        axes[0, 2].tick_params(axis="x", rotation=45)

        # 4. Train vs Test Split
        train_counts = self.train_df["Class Index"].map(self.class_names).value_counts()
        test_counts = self.test_df["Class Index"].map(self.class_names).value_counts()

        x = np.arange(len(self.class_names))
        width = 0.35

        axes[1, 0].bar(
            x - width / 2,
            train_counts.values,
            width,
            label="Train",
            color="lightblue",
            alpha=0.8,
        )
        axes[1, 0].bar(
            x + width / 2,
            test_counts.values,
            width,
            label="Test",
            color="lightcoral",
            alpha=0.8,
        )
        axes[1, 0].set_title(
            "Train vs Test Distribution", fontsize=14, fontweight="bold"
        )
        axes[1, 0].set_xlabel("News Category")
        axes[1, 0].set_ylabel("Number of Samples")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(list(self.class_names.values()), rotation=45)
        axes[1, 0].legend()

        # 5. Title vs Description Length Correlation
        scatter = axes[1, 1].scatter(
            self.full_df["title_length"],
            self.full_df["desc_length"],
            c=self.full_df["Class Index"],
            cmap="viridis",
            alpha=0.6,
            s=20,
        )
        axes[1, 1].set_title(
            "Title vs Description Length", fontsize=14, fontweight="bold"
        )
        axes[1, 1].set_xlabel("Title Length (characters)")
        axes[1, 1].set_ylabel("Description Length (characters)")
        plt.colorbar(scatter, ax=axes[1, 1], label="Class Index")

        # 6. Text Length by Class (Violin Plot)
        data_for_violin = []
        labels_for_violin = []
        for class_name in self.class_names.values():
            class_data = self.full_df[self.full_df["class_name"] == class_name][
                "text_length"
            ]
            data_for_violin.append(class_data)
            labels_for_violin.append(class_name)

        parts = axes[1, 2].violinplot(
            data_for_violin, positions=range(len(self.class_names)), showmeans=True
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(self.colors[i])
            pc.set_alpha(0.7)
        axes[1, 2].set_title(
            "Text Length Distribution by Class", fontsize=14, fontweight="bold"
        )
        axes[1, 2].set_xlabel("News Category")
        axes[1, 2].set_ylabel("Text Length (characters)")
        axes[1, 2].set_xticks(range(len(self.class_names)))
        axes[1, 2].set_xticklabels(list(self.class_names.values()), rotation=45)

        plt.tight_layout()
        plt.savefig("outputs/eda/overview_dashboard.png", dpi=300, bbox_inches="tight")
        plt.show()

    def create_word_clouds(self):
        print("\nCreating word clouds...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        english_stopwords = set(stopwords.words("english"))

        # Add additional common words and news-specific stop words

        # Combine all stop words
        comprehensive_stopwords = english_stopwords.union(additional_stopwords)

        for i, (class_idx, class_name) in enumerate(self.class_names.items()):
            class_text = " ".join(
                self.full_df[self.full_df["Class Index"] == class_idx]["text"]
            )

            class_text = re.sub(r"[^a-zA-Z\s]", "", class_text.lower())

            wordcloud = WordCloud(
                width=800,
                height=600,
                background_color="white",
                colormap="tab10",
                max_words=150,
                stopwords=comprehensive_stopwords,
                relative_scaling=0.5,
                min_font_size=10,
                collocations=False,  # Avoid repeated phrases
                min_word_length=3,  # Filter out very short words
            ).generate(class_text)

            axes[i].imshow(wordcloud, interpolation="bilinear")
            axes[i].set_title(
                f"{class_name} News - Key Terms", fontsize=16, fontweight="bold", pad=20
            )
            axes[i].axis("off")

        plt.suptitle(
            "Word Clouds by News Category", fontsize=20, fontweight="bold", y=0.95
        )
        plt.tight_layout()
        plt.savefig("outputs/eda/word_clouds.png", dpi=300, bbox_inches="tight")
        plt.show()

    def analyze_top_words(self):
        print("\nAnalyzing top words by class...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        english_stopwords = set(stopwords.words("english"))

        # Combine all stop words
        comprehensive_stopwords = english_stopwords.union(additional_stopwords)

        for i, (class_idx, class_name) in enumerate(self.class_names.items()):
            class_text = " ".join(
                self.full_df[self.full_df["Class Index"] == class_idx]["text"]
            )

            # Clean and count words
            words = re.findall(r"\b[a-zA-Z]{3,}\b", class_text.lower())
            word_counts = Counter(words)

            # Remove comprehensive stop words
            filtered_counts = {
                word: count
                for word, count in word_counts.items()
                if word not in comprehensive_stopwords
            }

            # Get top 15 words
            top_words = dict(Counter(filtered_counts).most_common(15))

            # Create horizontal bar chart
            y_pos = np.arange(len(top_words))
            axes[i].barh(
                y_pos, list(top_words.values()), color=self.colors[i], alpha=0.8
            )
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(list(top_words.keys()))
            axes[i].set_title(
                f"{class_name} - Top Words", fontsize=14, fontweight="bold"
            )
            axes[i].set_xlabel("Frequency")

            # Add value labels on bars
            for j, v in enumerate(top_words.values()):
                axes[i].text(
                    v + max(top_words.values()) * 0.01,
                    j,
                    str(v),
                    va="center",
                    fontweight="bold",
                )

        plt.suptitle(
            "Most Frequent Words by News Category", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig("outputs/eda/top_words_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def statistical_analysis(self):
        print("\nPerforming statistical analysis...")

        stats_by_class = (
            self.full_df.groupby("class_name")
            .agg(
                {
                    "text_length": ["mean", "std", "min", "max"],
                    "word_count": ["mean", "std", "min", "max"],
                    "title_length": ["mean", "std"],
                    "desc_length": ["mean", "std"],
                }
            )
            .round(2)
        )

        # Flatten column names
        stats_by_class.columns = [
            "_".join(col).strip() for col in stats_by_class.columns
        ]

        # Create a beautiful statistical summary plot with proper layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Text length statistics
        stats_data = []
        for class_name in self.class_names.values():
            class_data = self.full_df[self.full_df["class_name"] == class_name]
            stats_data.append(
                [
                    class_data["text_length"].mean(),
                    class_data["word_count"].mean(),
                    class_data["title_length"].mean(),
                    class_data["desc_length"].mean(),
                ]
            )

        stats_df = pd.DataFrame(
            stats_data,
            columns=["Text Length", "Word Count", "Title Length", "Desc Length"],
            index=list(self.class_names.values()),
        )

        # Heatmap of statistics
        sns.heatmap(
            stats_df.T,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            ax=axes[0, 0],
            cbar_kws={"label": "Average Value"},
        )
        axes[0, 0].set_title(
            "Average Text Statistics by Class", fontsize=14, fontweight="bold"
        )

        # Text length distribution
        data_by_class = [
            self.full_df[self.full_df["class_name"] == class_name]["text_length"]
            for class_name in self.class_names.values()
        ]
        axes[0, 1].boxplot(data_by_class, labels=list(self.class_names.values()))
        axes[0, 1].set_title(
            "Text Length Distribution by Class", fontsize=14, fontweight="bold"
        )
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Word count distribution
        data_by_class = [
            self.full_df[self.full_df["class_name"] == class_name]["word_count"]
            for class_name in self.class_names.values()
        ]
        axes[0, 2].boxplot(data_by_class, labels=list(self.class_names.values()))
        axes[0, 2].set_title(
            "Word Count Distribution by Class", fontsize=14, fontweight="bold"
        )
        axes[0, 2].tick_params(axis="x", rotation=45)

        # Correlation matrix
        corr_features = [
            "text_length",
            "word_count",
            "title_length",
            "desc_length",
            "Class Index",
        ]
        corr_matrix = self.full_df[corr_features].corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            ax=axes[1, 0],
        )
        axes[1, 0].set_title(
            "Feature Correlation Matrix", fontsize=14, fontweight="bold"
        )

        # Class balance visualization
        class_counts = self.full_df["class_name"].value_counts()
        axes[1, 1].bar(
            class_counts.index, class_counts.values, color=self.colors, alpha=0.8
        )
        axes[1, 1].set_title("Class Distribution", fontsize=14, fontweight="bold")
        axes[1, 1].set_ylabel("Number of Samples")
        axes[1, 1].tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for i, v in enumerate(class_counts.values):
            axes[1, 1].text(
                i,
                v + max(class_counts.values) * 0.01,
                str(v),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Title vs Description length scatter
        scatter = axes[1, 2].scatter(
            self.full_df["title_length"],
            self.full_df["desc_length"],
            c=self.full_df["Class Index"],
            cmap="viridis",
            alpha=0.6,
            s=10,
        )
        axes[1, 2].set_title(
            "Title vs Description Length", fontsize=14, fontweight="bold"
        )
        axes[1, 2].set_xlabel("Title Length (characters)")
        axes[1, 2].set_ylabel("Description Length (characters)")
        plt.colorbar(scatter, ax=axes[1, 2], label="Class Index")

        plt.suptitle("Statistical Analysis Summary", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            "outputs/eda/statistical_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        print("\nSummary Statistics by Class:")
        print(stats_by_class)

        return stats_by_class

    def topic_modeling_visualization(self):
        print("\nPerforming topic modeling...")

        sample_df = self.full_df.sample(n=3000, random_state=42)

        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
        )

        tfidf_matrix = vectorizer.fit_transform(sample_df["text"])

        # Perform LDA
        n_topics = 6
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=20
        )
        lda.fit(tfidf_matrix)

        # Get top words for each topic
        feature_names = vectorizer.get_feature_names_out()

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = topic[top_words_idx]

            # Create bar chart for each topic
            axes[topic_idx].barh(
                range(len(top_words)),
                top_weights,
                color=self.colors[topic_idx % len(self.colors)],
                alpha=0.8,
            )
            axes[topic_idx].set_yticks(range(len(top_words)))
            axes[topic_idx].set_yticklabels(top_words)
            axes[topic_idx].set_title(
                f"Topic {topic_idx + 1}", fontsize=14, fontweight="bold"
            )
            axes[topic_idx].set_xlabel("Weight")

        plt.suptitle(
            "Topic Modeling - Discovered Topics", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig("outputs/eda/topic_modeling.png", dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Identified {n_topics} topics from the text data")

    def run_complete_eda(self):
        self.load_data()

        self.create_overview_dashboard()
        self.create_word_clouds()
        self.analyze_top_words()
        self.statistical_analysis()
        self.topic_modeling_visualization()

        print("EDA completed successfully!")
        # blue color print
        print("\033[94mOutput stored in outputs/eda/\033[0m")


def main():
    eda = AGNewsEDA()
    eda.run_complete_eda()


if __name__ == "__main__":
    main()
