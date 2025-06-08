

# Load necessary libraries
library(ggplot2)
library(dplyr)
library(extrafont)  # For font control similar to matplotlib

# Load Arial font if available
# font_import()  # Uncomment if Arial isn't already loaded
loadfonts()    # Uncomment if needed

# Data
emmeans_df <- data.frame(
  used_story = c(FALSE, TRUE, FALSE, TRUE),
  quartile_type = c("Least active", "Least active", "Newcomers", "Newcomers"),
  prob = c(0.215, 0.288, 0.169, 0.247),
  asymp.LCL = c(0.203, 0.272, 0.159, 0.234),
  asymp.UCL = c(0.228, 0.304, 0.179, 0.261)
)

# Convert used_story to factor with labels (simplified to match template)
emmeans_df <- emmeans_df %>%
  mutate(used_story = factor(used_story, 
                             levels = c(FALSE, TRUE),
                             labels = c("Did not use \n personal narratives", 
                                        "Used personal \n narratives")))

# Define colors from Set2 palette to match template
set2_colors <- c("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854")
custom_colors <- c("Least active" = set2_colors[2],  # Second color from Set2 (#fc8d62)
                   "Newcomers" = set2_colors[3])      # Third color from Set2 (#8da0cb)

# Create the plot
p <- ggplot(emmeans_df, aes(x = used_story, y = prob, color = quartile_type, group = quartile_type)) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 5) +  # Slightly larger points for visibility
  # Add error bars for confidence intervals
  geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL), width = 0.3, linewidth = 1.2) +
  # Add text labels above points for probability values
  geom_text(aes(label = paste0(round(prob * 100, 2), "%")),
            hjust = 0.5, vjust = -1.0, size = 10, color = "black") +
  scale_color_manual(values = custom_colors) +
  # Fine-grain control over y-axis ticks and formatting
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    breaks = seq(0, 0.35, by = 0.025)
  ) +
  labs(
    x = NULL, 
    y = "Mean probability of returning next month", 
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
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),               # No minor grid
    axis.line = element_line(color = "black"),        # Add axis lines
    
    # Remove top and right spines for a cleaner look matching template
    panel.border = element_blank(),
    axis.line.x = element_line(color = "black"),
    axis.line.y = element_line(color = "black"),
    
    # Move legend to top right inside the plot
    legend.position = c(0.6, 0.99),
    legend.justification = c(1, 1),
    legend.background = element_rect(fill = "white", color = "gray90"),
    legend.margin = margin(6, 6, 6, 6),
    legend.title.align = 0.5,
    legend.key.size = unit(1.5, "lines"),
    
    # Apply the same figure size proportions
    plot.margin = margin(20, 20, 20, 20)
  )

# Print the plot
print(p)

# Save the plot with the same dimensions as template
ggsave("h3.png", plot = p, width = 14, height = 10, units = "in", dpi = 600)