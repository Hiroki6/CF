#include "CF.h"
#include "Cf.cpp"
#include <time.h>

using namespace std;

int main(){

  Item items;
  string item_name;
  int score;
  string user_name;
  int user_count = 0;
  // トップレコメンデーションnのユーザー配列
  vector<TopNUser> similar_users;
  // 現在のユーザー
  string current_user;
  // トップ何人かを示す変数
  int topN;
  // ユーザーIDの配列
  vector<string> userids;
  // ratings
  vector<Ratings> ratings;
  // ユーザーファイルと評価ファイルのパス
  string user_filepass("data/ml-1m/users.dat");
  string rating_filepass("data/ml-1m/ratings.dat");
  // criticsの作成
  Prefs critics;
  clock_t start, end;

  userids = create_userids(user_filepass);
  ratings = create_ratings(rating_filepass);
  critics = create_prefs(userids, ratings);
  
  /*cout<<"現在のユーザー:";
  cin>>current_user;
  cout<<"何人の類似ユーザーを表示しますか:";
  cin>>topN;
  
  similar_users = topMatches(critics, current_user, topN);
  cout<<"トップ"<<current_user<<"人の出力";
  PrintTopN(similar_users); */

  //PrintVector(userids);
  //PrintRatings(ratings);
  //PrintPrefs(critics);
  //cout<<critics.size();
  start = clock();
  similar_users = topMatches(critics, "3", 5);
  end = clock();
  cout<<(double)(end-start)/CLOCKS_PER_SEC<<endl;
  PrintTopN(similar_users);
  return 0;
}


