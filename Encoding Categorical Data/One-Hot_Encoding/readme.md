<h1>One-Hot Encoding</h1>
<p>
## 📖 Overview
**One-Hot Encoding** is a technique used in machine learning and data preprocessing to convert categorical variables into a format that can be provided to machine learning algorithms to improve their performance. Each category in a feature is transformed into a binary vector, with a unique position assigned for each category.

For example::

| Color | Red | Green | Blue |
|-------|-----|-------|------|
| Red   | 1   | 0     | 0    |
| Green | 0   | 1     | 0    |
| Blue  | 0   | 0     | 1    |

This technique is widely used to handle categorical data that has no ordinal relationship.

---

## 🔑 Key Concepts

### 1. **Categorical Variables**
Categorical data is non-numeric and falls into two types:
- **Nominal**: Categories have no inherent order (e.g., Gender, Color).
- **Ordinal**: Categories have a meaningful order (e.g., Education Levels: High School < Undergraduate < Graduate).

### 2. **Why One-Hot Encoding?**
- Machine learning models work better with numerical data.
- It eliminates the assumption of ordinal relationships in nominal data.
- Helps represent categorical variables in a mathematically consistent way.

---

## 🚧 How One-Hot Encoding Works

1. Identify the categorical variable in the dataset.
2. Create a binary column for each unique category.
3. Assign `1` to the binary column corresponding to the category of the sample and `0` to the others.

### Example:
#### Original Dataset:
| Color |
|-------|
| Red   |
| Green |
| Blue  |

#### After One-Hot Encoding:
| Red | Green | Blue |
|-----|-------|------|
| 1   | 0     | 0    |
| 0   | 1     | 0    |
| 0   | 0     | 1    |

---

## 🛑 The Dummy Variable Trap

The **Dummy Variable Trap** occurs when one-hot encoding results in redundancy. For \( n \) categories, the \( n \)-th column can be predicted from the others. This can lead to multicollinearity in models like linear regression.

### Example:
For three categories (`Red`, `Green`, `Blue`), if two binary columns are present:
- Red: \( 1 \) implies \( Green \) and \( Blue \) are \( 0 \).
- This redundancy creates a dependency among the columns.

#### Solution:
Drop one of the columns after encoding to avoid redundancy.

#### Modified Dataset:
| Red | Green |
|-----|-------|
| 1   | 0     |
| 0   | 1     |
| 0   | 0     |

  
</p>


<h1>Author-Aadit Sharma Shiwakoti</h1>
