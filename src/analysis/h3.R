library("lme4")
library("emmeans")
library("glmmTMB")
library(lmerTest)
library(dplyr)
library("stargazer")

df3 <- read.csv("../../data/h3.csv")
df3 <- df3[df3$month !=12,]
df3$month <- as.factor(df3$month)

df3$used_story <- df3$num_predictions_true >= 1

result <- df3 %>%
  group_by(month) %>%
  summarise(count = n(),  mean_value = mean(next_month_active_any))

# Print the result
print(result)


h5_model <- glmer( next_month_active_any ~ used_story*quartile_type +  (1|month), data=df3,family="binomial",
                   control=glmerControl(optimizer = "nloptwrap", calc.derivs = FALSE))

summary(h5_model)

stargazer(h5_model, type = "text", title = "Generalized Linear Mixed Model Results", 
          out = "glmm_results_h3.tex", digits = 3)


em_h5 <- emmeans(h5_model, pairwise ~ used_story*quartile_type,  pbkrtest.limit = 6000, lmerTest.limit = 6000,type = "response")

summary(em_h5)