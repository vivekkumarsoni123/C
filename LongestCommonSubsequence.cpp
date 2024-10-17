#include<bits/std++.h>
int func(int i,int j, string s1, string s2,vector<vector<int>>& dp) {
    
    if(i==n || j==m) return 0;
    if(dp[i][j] != -1) return dp[i][j];
    
    if(s1[i]==s2[j]) {
        return dp[i][j] = 1+ func(i+1,j+1,s1,s2,dp);
    }
    else {
        return dp[i][j] = max(func(i+1,j,s1,s2,dp),func(i,j+1,s1,s2,dp));
    }
}

int LongestCommonString(string s1, string s2) {
    int n = s1.size(), m = s2.size();
    vector<vector<int>>& dp(n,vector<int>(m,0));
    return func(0,0,s1,s2,dp);
}
