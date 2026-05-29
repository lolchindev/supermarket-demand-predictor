import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv(r"D:\Coding & extra\VS Code~\IP\superstore_final_dataset.csv")

landmarks = {'📈Stock UP' : '#46E809', '✅Normal' : '#219DFF', '📉Keep LOW' : '#E80909'}

def load_clean():
    df1 = df[['Row_ID', 'Order_Date', 'Country', 'City', 'Region', 'Category', 'Sub_Category', 'Sales']]
    df1['Order_Date'] = pd.to_datetime(df['Order_Date'], format = 'mixed', dayfirst = True)
    df1['Sales'] = np.log1p(df1['Sales'])
    df1['Year'] = df1['Order_Date'].dt.year
    df1['Month'] = df1['Order_Date'].dt.month
    df1['Month_Name'] = df1['Order_Date'].dt.strftime('%B')
    return df1

def prediction(df1, month: int, category: str, sub_category: str):
    #
    diff = (df1['Category'] == category) & (df1['Sub_Category'] == sub_category)
    subset = df1[diff].copy()

    #mm = month
    same_mm = subset[subset['Month'] == month]['Sales']
    avg_mm = same_mm.mean() if len(same_mm) else subset['Sales'].mean()

    monthly = (subset.groupby(['Year', 'Month'])['Sales'].mean().reset_index().sort_values(['Year', 'Month']))

    if len(monthly) >= 3:
        x = np.arange(len(monthly))
        #
        m, c = np.polyfit(x, monthly['Sales'].values, 1)
        trend = m * len(monthly) + c

    else:
        trend = avg_mm

    predicted = 0.6 * avg_mm + 0.4 * trend
    predicted = max(0, predicted)
    needed = predicted * 1.10

    #
    avg_last3 = subset.tail(90)['Sales'].mean()
    if predicted >= avg_last3 * 1.12:
        label = '📈Stock UP'
    elif predicted <= avg_last3 * 0.88:
        label = '📉Keep LOW'
    else:
        label = '✅Normal'

    final = {
        'Sub_Category':sub_category,
        'Avg_Past_Sale' : round(avg_mm, 4),
        'Trend_Value' : round(trend, 4),
        'Predicted_Sales' : round(predicted, 4),
        'Buffered_Sales' : round(needed, 4),
        'Recommendation' : label,
        'Orders_In_Date' : len(same_mm)
    }

    region_row = []
    for region, grp in subset.groupby('Region'):
        rm = grp[grp['Month'] == month]['Sales']
        avg_r = rm.mean() if len(rm) else grp['Sales'].mean()
        r_rows = []
        for (y, t), g in grp.groupby(['Year', 'Month']):
            r_rows.append(g['Sales'].mean())
        if len(r_rows) >= 3:
            rx =np.arange(len(r_rows))
            rs, ri =np.polyfit(rx, r_rows, 1)
            r_trend = rs * len(r_rows) + ri
        else:
            r_trend = avg_r
        r_pre =max(0, 0.6 * avg_r + 0.4 * r_trend) * 1.10
        r_lastmm =grp.tail(30)['Sales'].mean()
        if r_pre >= r_lastmm * 1.12: r_label = '📈Stock UP'
        elif r_pre <= r_lastmm * 0.88: r_label = '📉Keep LOW'
        else: r_label = '✅Normal'
        region_row.append({'Region': region, 'Predicted_Sales': round(r_pre, 4), 'Recommendation': r_label})

    region_df = pd.DataFrame(region_row)

    #
    his_df =monthly.copy()
    his_df['Period'] = his_df['Year'].astype(str) + '-' + his_df['Month'].astype(str).str.zfill(2)

    return final, region_df, his_df, subset

#
def dashboard(his_df, region_df, final, sub_category):
    plt.style.use('dark_background')

    fig, (ax1, ax2)= plt.subplots(1, 2, figsize = (14, 5))

    fig.suptitle(
        f'Demand Prediction: {sub_category}\n'
        f'Recommendation: {final['Recommendation']}\n'
        f'Predicted Sales ($ in millions): {final['Predicted_Sales']}\n'
        f'Buffer Stock ($ in millions): {final['Buffered_Sales']}',
        fontsize = 14, fontweight = 'bold', color = '#6BB2CF'
    )

    #
    ax1.plot(his_df['Period'], his_df['Sales'], marker = 'o', color = '#CFC444', label = 'Past Sales ($ in millions)')
    ax1.axhline(final['Predicted_Sales'], color = '#8F3700', linestyle = '--', label = 'Prediction Level')
    ax1.set_title('Historical Trend', fontsize = 12)
    ax1.set_ylabel('Sales')
    ax1.tick_params(axis = 'x', rotation = 45)
    ax1.legend()

    #
    ax2.bar(region_df['Region'], region_df['Predicted_Sales'], color = '#4622E3')
    ax2.set_title('Predicted Sales by Region', fontsize = 12)
    ax2.set_ylabel('Sales')

    #
    plt.tight_layout()

    #
    clean = sub_category.replace(' ', '_')
    file_n = f'prediction_{clean}.png'

    plt.savefig(file_n, bbox_inches = 'tight', dpi = 300)

    plt.close()

    print(f'✅ Dashboard successfully saved as: {file_n}')

    return file_n

def clear():
    os.system('cls' if os.name == 'nt' else 'clear') #Cleaning the terminal for the output to be perfectly placed

def banner():
    print("\033[96m" + "═"*50)
    print("  DEMAND PREDICTION ENGINE")
    print("═"*50 + "\033[0m\n")

def pick(prompt, options):
    print(f'\033[93m{prompt}\033[0m')
    for i, opt in enumerate(options, 1):
        print(f'[{i}] {opt}')

    while True:
        try:
            choice = int(input('\n Select Number: ').strip())
            if 1 <= choice <= len(options):
                return options[choice-1]
        except ValueError:
            pass
        print(f'⚠️Invalid choice, try again.⚠️')

def summary(final, region_df):
    colors = {'📈Stock UP' : '\033[38;2;70;232;9m', '✅Normal': '\033[38;2;33;157;255m', '📉Keep LOW': '\033[38;2;232;9;9m'}

    rec = final['Recommendation']
    c = colors.get(rec, '\033[0m')

    print("\n\033[96m" + "─"*50)
    print(f"  PREDICTION SUMMARY: {final['Sub_Category']}")
    print("─"*50 + "\033[0m")

    print(f' Base Trend: {final['Trend_Value']:.2f}')
    print(f' Predicted Sales ($ in millions): {final['Predicted_Sales']:.2f}')
    print(f' Buffered (+10%) ($ in millions): \033[38;2;173;161;0m{final['Buffered_Sales']:.2f}\033[0m')
    print(f' Action Required: {c}{rec}\033[0m\n')

    print(' Regional Breakdown ($ in millions)')
    for _, row in region_df.iterrows():
        r_color = colors.get(row['Recommendation'], '\033[0m')
        print(f'   {row['Region']:<10} {row['Predicted_Sales']:.2f}  {r_color}{row['Recommendation']}\033[0m')

    print("\n\033[90m  *Note: Sales values are in log1p scale. Buffered adds 10% safety margin.\033[0m\n")

#Main Loop
def main():
    print("\n⏳ Loading and cleaning data...")
    df1 = load_clean()

    categories = sorted(df1['Category'].unique().tolist())
    sub_cat_map = {c: sorted(df1[df1['Category']==c]['Sub_Category'].unique().tolist()) for c in categories}

    MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December']

    while True:
        clear()
        banner()

        month_name = pick("Step 1 of 3 — Select MONTH:", MONTHS)
        target_month = MONTHS.index(month_name) + 1

        print()
        category = pick("Step 2 of 3 — Select CATEGORY:", categories)

        print()
        sub_category = pick("Step 3 of 3 — Select SUB-CATEGORY:", sub_cat_map[category])

        print("\n⚙️  Running prediction...\n")
        final, region_df, his_df, subset = prediction(df1, target_month, category, sub_category)

        summary(final, region_df)

        dashboard(his_df, region_df, final, sub_category)

        print("\033[96m" + "─"*50 + "\033[0m")
        again = input("  🔃  Run another prediction? [y/n]: ").strip().lower()
        if again != 'y':
            print("\n  \033[38;2;229;225;225mGoodbye! 👋\033[0m\n")
            break

if __name__ == "__main__":
    main()
