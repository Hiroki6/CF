#include <iostream>
#include <math.h>
#include <time.h>
#include "CF.h"

using namespace std;

float sim_distance(Prefs prefs, string person1, string person2){
 
  // 共通するアイテムの差の平方の合計
  float sum_of_squares = 0;

  /*ここを並列化したい*/
  for( map<string, int>::iterator p1 = prefs[person1].begin(); p1 != prefs[person1].end(); p1++){
    if( prefs[person2].find((*p1).first) != prefs[person2].end() ){
      sum_of_squares = pow(prefs[person1][(*p1).first] - prefs[person2][(*p1).first], 2);
    }
  }
  
  // 両者ともに評価しているものが一つもなければ0を返す
  if(sum_of_squares == 0){
    return 0;
  }else{
    return 1.0/(1+sum_of_squares);
  }
}

float sim_pearson(Prefs prefs, string person1, string person2){

  // 二人とも評価しているアイテムのリストを作る
  Item si;
  
  // 共通するアイテムの差の平方の合計
  float sum_of_squares = 0;

  // 合計値
  int sum1 = 0;
  int sum2 = 0;
  int sum1Sq = 0;
  int sum2Sq = 0;
  int pSum = 0;

  // 共通のアイテムの数
  int n = 0;

  // ピアソン相関係数のための分子と分母
  float num;
  float den;
  // ピアソン相関係数
  float pearson;

  /*ここを並列化したい*/
  for( map<string, int>::iterator p1 = prefs[person1].begin(); p1 != prefs[person1].end(); p1++){
    if( prefs[person2].find((*p1).first) != prefs[person2].end() ){
      //si.insert(map<string, int>::value_type( (*p1).first, 1) );
      n++;
      sum1 += prefs[person1][(*p1).first];
      sum2 += prefs[person2][(*p1).first];
      sum1Sq += pow(prefs[person1][(*p1).first],2);
      sum2Sq += pow(prefs[person2][(*p1).first],2);
      pSum += prefs[person1][(*p1).first] * prefs[person2][(*p1).first];
    }
  }
  
  //n = (unsigned int)si.size();
  
  // 両者ともに評価しているものが一つもなければ0を返す
  if(n == 0){
    return 0;
  }

  // すべての嗜好を合計する
  /* ここも並列化できる */
  /*
  for(  map<string, int>::iterator si_it = si.begin(); si_it != si.end(); si_it++){
      sum1 += prefs[person1][(*si_it).first];
      sum2 += prefs[person2][(*si_it).first];
      sum1Sq += pow(prefs[person1][(*si_it).first],2);
      sum2Sq += pow(prefs[person2][(*si_it).first],2);
      pSum += prefs[person1][(*si_it).first] * prefs[person2][(*si_it).first];
  }
  */
  num = pSum * 1.0 - (sum1 * sum2 * 1.0/ n);
  den = sqrt((sum1Sq*1.0 - pow(sum1,2)*1.0/n) * (sum2Sq*1.0 - pow(sum2,2)*1.0/n));

  pearson = num / den;
  
  return pearson;
}

vector<TopNUser> topMatches(Prefs prefs, string person, int n){
  vector<TopNUser> scores;
  float score;
  string user_name;
  clock_t start,end;
  // personとそれ以外の人物の名前と類似度を含むディクショナリを返す
  /* ここの処理も並列化 */
  for( map<string, Item>::iterator user = prefs.begin(); user != prefs.end(); user++){
    if( (*user).first == person){
      continue;
    }
    start = clock();
    score = sim_pearson(prefs, person, (*user).first);
    end = clock();
    cout<<(double)(end-start)/CLOCKS_PER_SEC<<endl;
    user_name = (*user).first;
    TopNUser input_user = {user_name, score};
    scores.push_back(input_user);
  }
  
  // scoresの大きい順に並べ替える(バブルソート)
  sort(scores.begin(), scores.end());
  
  return scores;
}

void PrintPrefs(Prefs prefs){

  for( map<string, Item>::iterator user = prefs.begin(); user != prefs.end(); user++){
    cout<<"ユーザー名:"<<(*user).first<<endl;
    for( map<string, int>::iterator item = prefs[(*user).first].begin(); item != prefs[(*user).first].end(); item++){
      cout<<"アイテム名:"<<(*item).first<<"\t"<<"スコア:"<<(*item).second<<endl;
    }
  }
  
}

void PrintTopN(vector<TopNUser> users){
  
  for(int i = 0; i < users.size(); i++){
    cout<<"("<<users[i].user_name<<","<<users[i].score<<")"<<endl;
  }
  
}

vector<RecItem> getRecommmendations(Prefs prefs, string person){

  ItemF totals;
  ItemF simSums;
  float sim;
  int person_index;
  vector<RecItem> rankings;
  
  for( map<string, Item>::iterator user = prefs.begin(); user != prefs.end(); user++){
    
    // 自分自身とは比較しない
    if( (*user).first == person){
      continue;
    }

    sim = sim_pearson(prefs, person, (*user).first);

    // 0以下のスコアを無視する
    if(sim < 0){
      continue;
    }

    for( map<string, int>::iterator other = prefs[(*user).first].begin(); other != prefs[(*user).first].end(); other++){
      // まだ評価されていないアイテムの得点のみの算出
      if( prefs[person].find( (*other).first ) == prefs[person].end() || prefs[person][(*other).first] == 0 ){
        totals.insert(map<string, float>::value_type( (*other).first, 0.0) );
        totals[(*other).first] += (*other).second * sim;
        simSums.insert(map<string, float>::value_type( (*other).first, 0.0) );
        simSums[(*other).first] += sim;
      }
    }
    
  }

  return rankings;
  
}

vector<string> split(string str, string delim){
  
  vector<string> items;
  size_t dlm_idx;
  
  if(str.npos == (dlm_idx = str.find_first_of(delim))){
    items.push_back(str.substr(0, dlm_idx));
  }
  
  while(str.npos != (dlm_idx = str.find_first_of(delim))){
    if(str.npos == str.find_first_not_of(delim)){
      break;
    }
    items.push_back(str.substr(0, dlm_idx));
    dlm_idx++;
    str = str.erase(0, dlm_idx);
    if(str.npos == str.find_first_of(delim) && "" != str){
      items.push_back(str);
      break;
    }
  }
  
  return items;
}

// ユーザーIDの配列作成
vector<string> create_userids(string file_pass){

  fstream user_file(file_pass);
  vector<string> user;
  vector<string> userlist;
  string one_user;
  string delim = "::";
  
  while(getline(user_file,one_user)){
    user = split(one_user, delim);
    // ムービーID
    userlist.push_back(user[0]);
  }
  return userlist;
}

vector<Ratings> create_ratings(string file_pass){

  fstream rating_file(file_pass);
  vector<string> rating;
  string delim = "::";
  string one_rate;
  vector<Ratings> ratings;
  
  while(getline(rating_file, one_rate)){
    rating = split(one_rate, delim);
    Ratings a = {rating[0], rating[2], stoi(rating[4]) };
    ratings.push_back(a);    
  }

  return ratings;
}

// vector配列の出力(string)
void PrintVector(vector<string> str){
  
  for(int i = 0; i < str.size(); i++){
    cout<<str[i];
  }
  
}

// ratingsの出力
void PrintRatings(vector<Ratings> ratings){

  for(int i = 0; i < ratings.size(); i++){
    cout<<"ユーザーID:"<<ratings[i].user_id<<"\t";
    cout<<"評価アイテムID:"<<ratings[i].item_id<<"\t";
    cout<<"スコア:"<<ratings[i].score<<"\t";
  }
  
}

// Prefs作成
Prefs create_prefs(vector<string> userlist, vector<Ratings> ratings){

  Prefs prefs;

  for(int i = 0; i < ratings.size(); i++){
    prefs[ratings[i].user_id][ratings[i].item_id] = ratings[i].score;
  }
  
  return prefs;
}
