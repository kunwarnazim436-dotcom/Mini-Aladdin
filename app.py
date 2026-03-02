import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# App ka Title aur Design
st.set_page_config(page_title="Mini-Aladdin App", layout="wide")
st.title("🧞‍♂️ Mini-Aladdin: Portfolio Risk Analyzer")
st.markdown("Yeh app aapke portfolio ka risk, diversification, aur best allocation calculate karta hai.")

# Sidebar - User Inputs ke liye
st.sidebar.header("Aapka Portfolio")
tickers_input = st.sidebar.text_input("Stocks daalein (comma se alag karein):", "AAPL, MSFT, GOOGL, 526570.BO")
initial_investment = st.sidebar.number_input("Total Investment Amount:", min_value=1000, value=100000)

# Jab user button click karega tabhi code aage badhega
if st.sidebar.button("Analyze Portfolio"):
    
    # User ke text ko list mein badalna
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    
    # Ek loading spinner dikhana
    with st.spinner("Market data fetch kiya ja raha hai... Kripya pratiksha karein ⏳"):
        try:
            # Data Fetch
            data = yf.download(tickers, start="2025-01-01", end="2026-03-01")['Close']
            daily_returns = data.pct_change().dropna()
            
            # --- 1. CORRELATION ANALYSIS ---
            st.subheader("1. Diversification Check (Correlation Matrix)")
            st.write("Neela (Blue) rang achha hai, iska matlab stocks ek sath nahi girenge.")
            correlation_matrix = daily_returns.corr()
            
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=0.5, ax=ax1)
            st.pyplot(fig1)

            # --- 2. MONTE CARLO SIMULATION ---
            st.subheader("2. Value at Risk (VaR) - Monte Carlo Simulation")
            weights_equal = np.array([1/len(tickers)] * len(tickers))
            portfolio_daily_returns = daily_returns.dot(weights_equal)

            days = 252 
            simulations = 1000 
            mu = portfolio_daily_returns.mean()
            sigma = portfolio_daily_returns.std()

            simulated_returns = np.random.normal(mu, sigma, (days, simulations))
            simulated_price_paths = initial_investment * np.cumprod(1 + simulated_returns, axis=0)

            final_portfolio_values = simulated_price_paths[-1, :] 
            var_95 = np.percentile(final_portfolio_values, 5)

            st.error(f"🚨 **Worst-Case Scenario (95% VaR):** 1 saal baad aapka portfolio girkar **{var_95:,.2f}** tak aa sakta hai.")
            st.success(f"💰 **Expected Average Value:** {np.mean(final_portfolio_values):,.2f}")

            # --- 3. PORTFOLIO OPTIMIZATION ---
            st.subheader("3. Portfolio Optimization (Efficient Frontier)")
            st.write("Aladdin 10,000 combinations test karke sabse best allocation nikal raha hai...")
            
            mean_returns = daily_returns.mean()
            cov_matrix = daily_returns.cov()

            num_portfolios = 10000
            results = np.zeros((3, num_portfolios))
            weights_record = []

            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights) 
                weights_record.append(weights)
                
                portfolio_return = np.sum(mean_returns * weights) * 252
                portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                
                results[0,i] = portfolio_return
                results[1,i] = portfolio_std_dev
                results[2,i] = portfolio_return / portfolio_std_dev 

            max_sharpe_idx = np.argmax(results[2])
            best_weights = weights_record[max_sharpe_idx]

            # Results ko do columns mein dikhana
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌟 Sabse Behtarin (Optimal) Portfolio")
                for i in range(len(tickers)):
                    st.write(f"👉 **{tickers[i]}:** {best_weights[i]*100:.2f}%")
            
            with col2:
                st.markdown("### 📊 Expected Results")
                st.write(f"**Expected Annual Profit:** {results[0,max_sharpe_idx]*100:.2f}%")
                st.write(f"**Expected Annual Risk:** {results[1,max_sharpe_idx]*100:.2f}%")

            # Efficient Frontier Graph
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            scatter = ax2.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
            plt.colorbar(scatter, label='Sharpe Ratio')
            ax2.scatter(results[1,max_sharpe_idx], results[0,max_sharpe_idx], marker='*', color='r', s=500, label='Best Portfolio')
            ax2.set_title('Mini-Aladdin: Efficient Frontier')
            ax2.set_xlabel('Risk')
            ax2.set_ylabel('Profit')
            ax2.legend()
            st.pyplot(fig2)

            st.balloons() # Success hone par thoda animation!

        except Exception as e:
            st.error(f"Ek error aayi hai. Kripya check karein ki aapne stock tickers sahi likhe hain ya nahi. (Error: {e})")
