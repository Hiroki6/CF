#ifndef MEMORY_H
#define MEMORY_H


#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fstream>

using namespace std;

typedef map<string, int> Item;

typedef map<string, Item> Prefs;

// ユーザー名と類似スコア
struct TopNUser{
  string user_name;
  float score;

  bool operator<(const TopNUser& right) const{
    return score == right.score ? user_name > right.user_name : score > right.score;
  }
};

// アイテムを評価したユーザーID,アイテム名,評価を格納した構造体
struct Ratings{
  string user_id;
  string item_id;
  int score;
};

// 一人のユーザーに対するアイテムと推薦度
struct RecItem{
  string item_name;
  float score;
};

// 二人の人物の距離を基にした類似性スコアを返す関数
float sim_distance(Prefs prefs, string person1, string person2);

// 二人の人物のピアソン相関係数を返す関数
float sim_pearson(Prefs prefs, string person1, string person2);

// トップnのレコメンデーション
vector<TopNUser> topMatches(Prefs prefs, string person, int n);

// Prefsの内容を出力する
void PrintPrefs(Prefs prefs);

// topNの出力
void PrintTopN(vector<TopNUser> users);

// アイテム名と浮動小数点スコアのディクショナリ
typedef map<string, float> ItemF;

// person以外の全ユーザーの評点の重み付け平均を使い、personへの推薦を算出する
vector<RecItem> getRecommmendations(Prefs prefs, string person);

// 文字列分割関数
vector<string> split(string str, string delim);


// ユーザーIDの配列作成
vector<string> create_userids(string file_pass);

// ratingsの作成
vector<Ratings> create_ratings(string file_pass);

// vector配列の出力(string)
void PrintVector(vector<string> str);

// ratingsの出力
void PrintRatings(vector<Ratings> ratings);

// Prefs作成
Prefs create_prefs(vector<string> userlist,vector<Ratings> ratings);

#endif
