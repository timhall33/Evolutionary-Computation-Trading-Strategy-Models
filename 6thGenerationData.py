import pandas as pd
import talib



file_path = 'updated_file_2.csv'
data = pd.read_csv(file_path)

grouped = data.groupby(['Symbol', 'Date'])

morningStarList = []
hammerList = []
piercingList = []
invertedHammerList = []
threeWhiteSoldiersList = []
engulfingList = []
hamariList = []
beltHoldList = []
threeInsideUpList = []
kickerList = []
shootingStarList = []
darkCloudCoverList = []
eveningStarList = []
hangingManList = []
threeBlackCrowsList = []


for name, group in grouped:
    morningStar = talib.CDLMORNINGSTAR(group['Open'], group['High'], group['Low'], group['Close']) #bullish
    hammer = talib.CDLHAMMER(group['Open'], group['High'], group['Low'], group['Close']) #bullish
    piercing = talib.CDLPIERCING(group['Open'], group['High'], group['Low'], group['Close']) #bullish
    invertedHammer = talib.CDLINVERTEDHAMMER(group['Open'], group['High'], group['Low'], group['Close']) #bullish
    threeWhiteSoldiers = talib.CDL3WHITESOLDIERS(group['Open'], group['High'], group['Low'], group['Close']) #bullish
    

    engulfing = talib.CDLENGULFING(group['Open'], group['High'], group['Low'], group['Close']) # both
    hamari = talib.CDLHARAMI(group['Open'], group['High'], group['Low'], group['Close']) #both
    beltHold = talib.CDLBELTHOLD(group['Open'], group['High'], group['Low'], group['Close']) #both
    threeInsideUp = talib.CDL3INSIDE(group['Open'], group['High'], group['Low'], group['Close']) #both
    kicker = talib.CDLKICKINGBYLENGTH(group['Open'], group['High'], group['Low'], group['Close']) #both


    shootingStar = talib.CDLSHOOTINGSTAR(group['Open'], group['High'], group['Low'], group['Close']) #bearish
    darkCloudCover = talib.CDLDARKCLOUDCOVER(group['Open'], group['High'], group['Low'], group['Close']) #bearish
    eveningStar = talib.CDLEVENINGSTAR(group['Open'], group['High'], group['Low'], group['Close']) #bearish
    hangingMan = talib.CDLHANGINGMAN(group['Open'], group['High'], group['Low'], group['Close']) #bearish
    threeBlackCrows = talib.CDL3BLACKCROWS(group['Open'], group['High'], group['Low'], group['Close']) #bearish


    morningStarList.extend(morningStar)
    hammerList.extend(hammer)
    piercingList.extend(piercing)
    invertedHammerList.extend(invertedHammer)
    threeWhiteSoldiersList.extend(threeWhiteSoldiers)
    engulfingList.extend(engulfing)
    hamariList.extend(hamari)
    beltHoldList.extend(beltHold)
    threeInsideUpList.extend(threeInsideUp)
    kickerList.extend(kicker)
    shootingStarList.extend(shootingStar)
    darkCloudCoverList.extend(darkCloudCover)
    eveningStarList.extend(eveningStar)
    hangingManList.extend(hangingMan)
    threeBlackCrowsList.extend(threeBlackCrows)

data['morningStar'] = morningStarList
data['hammer'] = hammerList
data['piercing'] = piercingList
data['invertedHammer'] = invertedHammerList
data['threeWhiteSoldiers'] = threeWhiteSoldiersList
data['engulfing'] = engulfingList
data['hamari'] = hamariList
data['beltHold'] = beltHoldList
data['threeInsideUp'] = threeInsideUpList
data['kicker'] = kickerList
data['shootingStar'] = shootingStarList
data['darkCloudCover'] = darkCloudCoverList
data['eveningStarList'] = eveningStarList
data['hangingMan'] = hangingManList
data['threeBlackCrows'] = threeBlackCrowsList

data.to_csv('updated_file_3.csv', index=False)