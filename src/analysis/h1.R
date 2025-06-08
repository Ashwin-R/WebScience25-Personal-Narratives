# Load necessary libraries
library("lme4")
library("emmeans")
library("glmmTMB")
library("lmerTest")
library("stargazer")

df <- read.csv("final_h1.csv")
df$subreddit <- as.factor(df$subreddit)
df$month <- as.factor(df$month)
df$quartile <- as.factor(df$quartile)

# Relevel the quartile to set "top 25%" as the reference category
df$quartile <- relevel(df$quartile, ref = "top 25%")

# Fit the generalized linear mixed model
h1_model <- glmer(is_narrative ~ quartile + (1|subreddit) + (1|month) + 
                    is_top_level + is_corona + cbrt_post_score, 
                  data = df, 
                  family = "binomial", 
                  control = glmerControl(optimizer = "nloptwrap", calc.derivs = FALSE))

# Summarize the model
summary(h1_model)

# Calculate estimated marginal means
em_m1 <- emmeans(h1_model, pairwise ~ quartile, 
                 pbkrtest.limit = 6000, lmerTest.limit = 6000, type = "response")
em_m1


# Convert the model summary to LaTeX format using stargazer
stargazer(h1_model, type = "text", title = "Generalized Linear Mixed Model Results", 
          out = "glmm_results.tex", digits = 3)

# Save the estimated marginal means in a data frame for further use
emmeans_df <- as.data.frame(summary(em_m1))

# Output the estimated marginal means table using stargazer
stargazer(emmeans_df, type = "text", title = "Estimated Marginal Means", 
          summary = FALSE, out = "emmeans_results.tex")