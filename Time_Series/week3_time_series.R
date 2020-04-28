library(timeSeries)
library(forecast)

simple_ts<-ts(c(35,25,30,35,32))
is.ts(simple_ts)
# 평활상수값(smoothing) ,초기값은 관측값 그대로 쓰기, h인자로 예측할 값 개수 지정
simple_ts_fit<-ses(simple_ts, alpha = 0.2, initial='simple', h=3)
# 적합된(fitted) 값
simple_ts_fit$fitted
summary(simple_ts_fit)
setwd('C:\\Users\\joyh1\\Desktop\\빅데이터_20-1\\R_data')
ses_pra<-read.csv("ses_pra.csv")
ses_pra_ts<-ts(ses_pra$sales, start=c(1987,1), frequency = 1)
# 알파값 지정 x -> 최적값 알아서 찾아줌, initial='simple'이면 알파값 지정해줘야함!
ses_pra_fit<-ses(ses_pra_ts, initial='optimal',h=3)
ses_pra_fit
summary(ses_pra_fit)
# 단순지수평활법 적용한 데이터 그래프 시각화
plot(ses_pra_fit)
# 단순지수평활법 적용후 fitted된 데이터 그래프 선 추가로 그리기
lines(fitted(ses_pra_fit), type='o', col='red')

## 내장된 oil data로 해보기
library(fpp2)
oil_ts<-window(oil, start=1996)
# alpha=0.1일 때
oil_ts_fit<-ses(oil_ts, alpha=0.1,initial='simple', h=5)
summary(oil_ts_fit)
plot(oil_ts_fit)
lines(fitted(oil_ts_fit), type='o', col='red')
# alpha=0.5일 때 -> RMSE가 alpha=0.1일 때보다 낮음
oil_ts_fit<-ses(oil_ts, alpha=0.5, initial='simple', h=5)
summary(oil_ts_fit)
plot(oil_ts_fit)
lines(fitted(oil_ts_fit), type='o', col='red')
# alpha=0.9일 때 -> RMSE가 alpha=0.5일 때보다 높음
oil_ts_fit<-ses(oil_ts_fit, alpha=0.9, initial='simple', h=5)
summary(oil_ts_fit)
plot(oil_ts_fit)
lines(fitted(oil_ts_fit), type='o', col='red')
names(oil_ts_fit)

