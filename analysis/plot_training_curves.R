library(tidyverse)

# SETTINGS
source("common.R")

# DATA: TRAINING AND VALIDATION RECOVERY
curves <- read_csv("training_curves_val.csv", show_col_types = FALSE) %>%
  pivot_longer(
    cols = -step,
    names_to = c("checkpoint", "split"),
    names_pattern = "(patient|impatient)_(train|val)_u_rec",
    values_to = "recovery",
    values_drop_na = TRUE
  ) %>%
  mutate(
    checkpoint = factor(checkpoint, levels = c("patient", "impatient"), labels = c("Patient-trained", "Impatient-trained")),
    split = factor(split, levels = c("train", "val"), labels = c("Training", "Validation"))
  ) %>%
  arrange(split, checkpoint, step)

# PLOT: OBSERVED LEARNING CURVES
learning_curves <- curves %>%
  ggplot(aes(step, recovery, color = checkpoint)) +
  geom_line(linewidth = 0.4) +
  geom_point(data = curves %>% filter(split == "Validation"), size = 0.75) +
  facet_grid(cols = vars(split)) +
  scale_color_manual(values = model_colors) +
  scale_x_continuous(breaks = seq(0, 300, 100), limits = c(0, 300)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), labels = scales::label_percent()) +
  labs(x = "Training step", y = "Utility recovery") +
  guides(color = guide_legend(nrow = 1))
save_figure(learning_curves, "training_validation_curves", height = 2.8, width = 7)
