library(ggplot2)
library(dplyr)
library(extrafont)  # For font control similar to matplotlib

# Load Arial font if available (may need font_import() first if not already loaded)
font_import()  # Uncomment if Arial isn't already loaded
loadfonts()    # Uncomment if needed

# Construct a data frame from the provided results
plot_df <- data.frame(
  is_narrative = c(FALSE, TRUE, FALSE, TRUE, FALSE, TRUE),
  quartile_type = c("Most active", "Most active", "Least active", "Least active", "Newcomers", "Newcomers"),
  mean_score = c(3.04, 3.86, 1.55, 2.72, 1.27, 2.28),
  asymp.LCL = c(2.71, 3.42, 1.34, 2.38, 1.08, 1.99),
  asymp.UCL = c(3.40, 4.35, 1.78, 3.09, 1.47, 2.60)
)

# Convert is_narrative to a factor with descriptive labels
plot_df <- plot_df %>%
  mutate(
    is_narrative = factor(is_narrative, 
                         levels = c(FALSE, TRUE),
                         labels = c("Other comments", 
                                    "Personal narratives")),
    # Define the legend order as requested
    quartile_type = factor(quartile_type, 
                          levels = c("Least active", "Newcomers", "Most active"))
  )

# Define colors matching the Set2 palette used in the Python code
# Selecting colors to match with the Set2 palette
set2_colors <- c("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854")
custom_colors <- c("Most active" = set2_colors[1],  # First color from Set2
                   "Least active" = set2_colors[2],  # Second color from Set2
                   "Newcomers" = set2_colors[3])     # Third color from Set2

# Create the plot
p <- ggplot(plot_df, aes(x = is_narrative, y = mean_score, color = quartile_type, group = quartile_type)) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 5) +  # Slightly larger points for visibility
  # Add error bars for confidence intervals
  geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL), width = 0.3, linewidth = 1.2) +
  # Add text labels above points for mean scores
  geom_text(aes(label = round(mean_score, 2)),
            hjust = 0.5, vjust = -1.0, size = 10, color = "black") +
  scale_color_manual(values = custom_colors) +
  # Fine-grain control over y-axis ticks and formatting
  scale_y_continuous(
    breaks = seq(0, 5, by = 0.5)
  ) +
  labs(
    x = NULL, 
    y = "Mean score", 
    color = "User group"
  ) +
  theme_minimal() +  # Start with a minimal theme similar to whitegrid in Python
  theme(
    text = element_text(family = "sans", size = 30),  # General font setting
    axis.text.x = element_text(size = 30, margin = margin(t = 10)),            # X-axis tick labels
    axis.text.y = element_text(size = 30),            # Y-axis tick labels
    axis.title.x = element_text(size = 30, margin = margin(t = 15)),  # X-axis label
    axis.title.y = element_text(size = 30, margin = margin(r = 15)),  # Y-axis label
    legend.text = element_text(size = 30),            # Legend text
    legend.title = element_text(size = 30),           # Legend title
    legend.title.align = 0.5,
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),               # No minor grid
    axis.line = element_line(color = "black"),        # Add axis lines
    
    # Remove top and right spines for a cleaner look matching H1
    panel.border = element_blank(),
    axis.line.x = element_line(color = "black"),
    axis.line.y = element_line(color = "black"),
    
    # Move legend to top right inside the plot
    legend.position = c(0.6, 0.99),
    legend.justification = c(1, 1),
    legend.background = element_rect(fill = "white", color = "gray90"),
    legend.margin = margin(6, 6, 6, 6),
    legend.key.size = unit(1.5, "lines"),
    
    # Apply the same figure size proportions
    plot.margin = margin(20, 20, 20, 20)
  )

# Print the plot
print(p)

# Save the plot with the same dimensions as H1
ggsave("RQ.png", plot = p, width = 14, height = 10, units = "in", dpi = 600)