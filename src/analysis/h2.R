# Load necessary libraries
library("lme4")
library("emmeans")
library("glmmTMB")
library("lmerTest")
library("stargazer")

# Read and prepare the data
df2 <- read.csv("../../data/h2.csv")
df2$subreddit <- as.factor(df2$subreddit)
df2$month <- as.factor(df2$month)

# Fit the generalized linear mixed model
h2_model <- glmer(
  cbind(num_low_users, num_unique_users - num_low_users) ~ # Instead of num_low_quartile - num_low_users, user_type column tells you the quartile of the users (is_narrative * user_type) interaction effect
    is_narrative*quartile_type + 
    (1 | subreddit) + 
    (1 | month) + 
    is_top_level + 
    is_corona + 
    cbrt_post_score + 
    cbrt_comment_score, 
  data = df2, 
  family = "binomial", 
  control = glmerControl(optimizer = "nloptwrap", calc.derivs = FALSE)
)

# Summarize the model
summary(h2_model)

# Calculate estimated marginal means
em_m2 <- emmeans(h2_model, pairwise ~ is_narrative*quartile_type, 
                 pbkrtest.limit = 6000, 
                 lmerTest.limit = 6000, 
                 type = "response")
em_m2
# Convert the model summary to LaTeX format using stargazer
stargazer(h2_model, type = "text", title = "Generalized Linear Mixed Model Results", 
          out = "glmm_results_h2.tex", digits = 3)

# Save the estimated marginal means in a data frame for further use
emmeans_df <- as.data.frame(summary(em_m2))

# Output the estimated marginal means table using stargazer
stargazer(emmeans_df, type = "text", title = "Estimated Marginal Means", 
          summary = FALSE, out = "emmeans_results_h2.tex")


# Fit the generalized linear mixed model
h2_model <- glmer(
  cbind(num_new_users, num_unique_users - num_new_users) ~ 
    is_narrative + 
    (1 | subreddit) + 
    (1 | month) + 
    is_top_level + 
    is_corona + 
    cbrt_post_score + 
    cbrt_comment_score, 
  data = df2, 
  family = "binomial", 
  control = glmerControl(optimizer = "nloptwrap", calc.derivs = FALSE)
)

# Summarize the model
summary(h2_model)

# Calculate estimated marginal means
em_m2 <- emmeans(h2_model, pairwise ~ is_narrative, 
                 pbkrtest.limit = 6000, 
                 lmerTest.limit = 6000, 
                 type = "response")

# Convert the model summary to LaTeX format using stargazer
stargazer(h2_model, type = "text", title = "Generalized Linear Mixed Model Results", 
          out = "glmm_results_h2_a.tex", digits = 3)

# Save the estimated marginal means in a data frame for further use
emmeans_df <- as.data.frame(summary(em_m2))

# Output the estimated marginal means table using stargazer
stargazer(emmeans_df, type = "text", title = "Estimated Marginal Means", 
          summary = FALSE, out = "emmeans_results_h2_b.tex")