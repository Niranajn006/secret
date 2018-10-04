Sub update_SCBSFF()

Portname = "SCBSFF"

Application.Calculation = xlCalculationManual
Sheets("DS CashProjection").Select
Sheets("DS CashProjection").Cells.ClearContents
Call scbam.RetrieveDataSet("DDE_CashProjection", "Portfolio  S", Portname, "=")
Sheets("DS CashProjection").Range("A1:BC1000").Sort Key1:=Range("J2"), Order1:=xlAscending, Header:= _
        xlGuess, OrderCustom:=1, MatchCase:=False, Orientation:=xlTopToBottom, _
        DataOption1:=xlSortNormal

Sheets(Portname).Calculate
date_col = Sheets(Portname).Range("A3").Value - 1

'Clear Propose
Sheets(Portname).Range("A45").Offset(0, date_col).ClearContents

'Data from registra
For i = 0 To 1
Sheets("Main").Range("B3").Value = Sheets(Portname).Range("A5").Offset(0, date_col + i).Value
Sheets("Main").Range("B4").Calculate
ii = Sheets("Main").Range("B4").Value
'Application (BN)
'If Sheets("DS CashProjection").Range("U3").Offset(ii, 0).Value > Sheets(Portname).Range("A25").Offset(0, date_col + i).Value Then
'Sheets(Portname).Range("A24").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("U3").Offset(ii, 0).Value - Sheets(Portname).Range("A25").Offset(0, date_col + i).Value
'Else
'Sheets(Portname).Range("A24").Offset(0, date_col + i).Value = 0
'End If
Sheets(Portname).Range("A24").Offset(0, date_col + i).Value = Sheets(Portname).Range("A24").Offset(0, date_col + i).Value
'Application (Not BN): Do nothing
Sheets(Portname).Range("A25").Offset(0, date_col + i).Value = Sheets(Portname).Range("A25").Offset(0, date_col + i).Value
'Switch-in (T+1)
Sheets(Portname).Range("A26").Offset(0, date_col + i).Value = Sheets(Portname).Range("A26").Offset(0, date_col + i).Value

'Redemption
Sheets(Portname).Range("A38").Offset(0, date_col + i).Value = Sheets(Portname).Range("A38").Offset(0, date_col + i).Value
'Switch-out
Sheets(Portname).Range("A39").Offset(0, date_col + i).Value = Sheets(Portname).Range("A39").Offset(0, date_col + i).Value
Next i


'From BOS
For i = 0 To 14
Sheets("Main").Range("B3").Value = Sheets(Portname).Range("A5").Offset(0, date_col + i).Value
Sheets("Main").Range("B4").Calculate
If IsError(Sheets("Main").Range("B4").Value) Then
Exit For
End If
ii = Sheets("Main").Range("B4").Value

'Debt Mature & Sell (BN)
Sheets(Portname).Range("A28").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("AJ3").Offset(ii, 0).Value
'Residual from Debt sell Bahtnet
If Sheets("DS CashProjection").Range("O3").Offset(ii, 0).Value > Sheets("DS CashProjection").Range("AJ3").Offset(ii, 0).Value Then
Sheets(Portname).Range("A29").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("O3").Offset(ii, 0).Value - Sheets("DS CashProjection").Range("AJ3").Offset(ii, 0).Value
Else
Sheets(Portname).Range("A29").Offset(0, date_col + i).Value = 0
End If
'CashEqSell_BahtNet
Sheets(Portname).Range("A30").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("AL3").Offset(ii, 0).Value
'Residual from Debt sell Bahtnet
If Sheets("DS CashProjection").Range("R3").Offset(ii, 0).Value > Sheets("DS CashProjection").Range("AL3").Offset(ii, 0).Value Then
Sheets(Portname).Range("A31").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("R3").Offset(ii, 0).Value - Sheets("DS CashProjection").Range("AL3").Offset(ii, 0).Value
Else
Sheets(Portname).Range("A31").Offset(0, date_col + i).Value = 0
End If
'Coupon on Bond
Sheets(Portname).Range("A32").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("P3").Offset(ii, 0).Value
'Interest on Deposit
Sheets(Portname).Range("A33").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("S3").Offset(ii, 0).Value
'Debt Buy (BN)
Sheets(Portname).Range("A40").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("AI3").Offset(ii, 0).Value
'Debt Buy (Not BN)
If Sheets("DS CashProjection").Range("N3").Offset(ii, 0).Value > Sheets("DS CashProjection").Range("AI3").Offset(ii, 0).Value Then
Sheets(Portname).Range("A41").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("N3").Offset(ii, 0).Value - Sheets("DS CashProjection").Range("AI3").Offset(ii, 0).Value
Else
Sheets(Portname).Range("A41").Offset(0, date_col + i).Value = 0
End If

'Deposit (BN)
Sheets(Portname).Range("A42").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("AK3").Offset(ii, 0).Value
'Deposit (Not BN)
If Sheets("DS CashProjection").Range("R3").Offset(ii, 0).Value > Sheets("DS CashProjection").Range("AK3").Offset(ii, 0).Value Then
Sheets(Portname).Range("A43").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("Q3").Offset(ii, 0).Value - Sheets("DS CashProjection").Range("AK3").Offset(ii, 0).Value
Else
Sheets(Portname).Range("A43").Offset(0, date_col + i).Value = 0
End If
'Expense
Sheets(Portname).Range("A44").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("T3").Offset(ii, 0).Value

'CashBal
Sheets(Portname).Range("A50").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("Y3").Offset(ii, 0).Value
'CashForUse
Sheets(Portname).Range("A53").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("AQ3").Offset(ii, 0).Value
'CashForBahtNet
Sheets(Portname).Range("A55").Offset(0, date_col + i).Value = Sheets("DS CashProjection").Range("AR3").Offset(ii, 0).Value

If i >= 2 Then
'Calculate deals
Sheets(Portname).Range("A28").Offset(0, date_col + i).FormulaR1C1 = _
        "=" & Sheets(Portname).Range("A28").Offset(0, date_col + i).Value & "+SUMIFS(Main!R1C34:R502C34,Main!R1C17:R502C17,""=" & Portname & """,Main!R1C19:R502C19,""=""&" & Portname & "!R5C,Main!R1C35:R502C35,""=Bahtnet"",Main!R1C20:R502C20,""=S"")"
Sheets(Portname).Range("A29").Offset(0, date_col + i).FormulaR1C1 = _
        "=" & Sheets(Portname).Range("A29").Offset(0, date_col + i).Value & "+SUMIFS(Main!R1C34:R502C34,Main!R1C17:R502C17,""=" & Portname & """,Main!R1C19:R502C19,""=""&" & Portname & "!R5C,Main!R1C35:R502C35,""=Cheque"",Main!R1C20:R502C20,""=S"")"
Sheets(Portname).Range("A40").Offset(0, date_col + i).FormulaR1C1 = _
        "=" & Sheets(Portname).Range("A40").Offset(0, date_col + i).Value & "+SUMIFS(Main!R1C34:R502C34,Main!R1C17:R502C17,""=" & Portname & """,Main!R1C19:R502C19,""=""&" & Portname & "!R5C,Main!R1C35:R502C35,""=Bahtnet"",Main!R1C20:R502C20,""=B"")"
Sheets(Portname).Range("A41").Offset(0, date_col + i).FormulaR1C1 = _
        "=" & Sheets(Portname).Range("A41").Offset(0, date_col + i).Value & "+SUMIFS(Main!R1C34:R502C34,Main!R1C17:R502C17,""=" & Portname & """,Main!R1C19:R502C19,""=""&" & Portname & "!R5C,Main!R1C35:R502C35,""=Cheque"",Main!R1C20:R502C20,""=B"")"
End If

Next i

'Look into Buy-Sell file
Sheets(Portname).Range("A24").Offset(0, date_col + 2).FormulaR1C1 = "=VLOOKUP(R4C1,'Cash intraday'!C27:C32,2,FALSE)"
Sheets(Portname).Range("A25").Offset(0, date_col + 2).FormulaR1C1 = "=VLOOKUP(R4C1,'Cash intraday'!C27:C32,3,FALSE)"
Sheets(Portname).Range("A26").Offset(0, date_col + 2).FormulaR1C1 = "=VLOOKUP(R4C1,'Cash intraday'!C27:C32,4,FALSE)"
Sheets(Portname).Range("A27").Offset(0, date_col + 2).FormulaR1C1 = 0
Sheets(Portname).Range("A38").Offset(0, date_col + 2).FormulaR1C1 = "=-VLOOKUP(R4C1,'Cash intraday'!C27:C32,5,FALSE)"
Sheets(Portname).Range("A39").Offset(0, date_col + 2).FormulaR1C1 = "=-VLOOKUP(R4C1,'Cash intraday'!C27:C32,6,FALSE)"

 'Shading
    Sheets(Portname).Range("A24:A27").Offset(0, date_col + 1).Interior.Pattern = xlNone
    Sheets(Portname).Range("A38:A39").Offset(0, date_col + 1).Interior.Pattern = xlNone
    Sheets(Portname).Range("A24:A27").Offset(0, date_col + 2).Interior.Color = 65535
    Sheets(Portname).Range("A38:A39").Offset(0, date_col + 2).Interior.Color = 65535

'Deposit
    Sheets(Portname).Range("A30").Offset(0, date_col + 2).FormulaR1C1 = _
        "=IF(R[-18]C<0,-R[-18]C,0) + IF(R[-15]C<0,-R[-15]C,0) + IF(R[-12]C<0,-R[-12]C,0) + " & Sheets(Portname).Range("A30").Offset(0, date_col + 2).Value
    Sheets(Portname).Range("A42").Offset(0, date_col + 2).FormulaR1C1 = _
        "=IF(R[-30]C>0,R[-30]C,0) + IF(R[-27]C>0,R[-27]C,0) + IF(R[-24]C>0,R[-24]C,0) + " & Sheets(Portname).Range("A42").Offset(0, date_col + 2).Value
    Sheets(Portname).Range("A30").Offset(0, date_col + 1).FormulaR1C1 = _
        "=IF(R[-18]C<0,-R[-18]C,0) + IF(R[-15]C<0,-R[-15]C,0) + IF(R[-12]C<0,-R[-12]C,0) + " & Sheets(Portname).Range("A30").Offset(0, date_col + 1).Value
    Sheets(Portname).Range("A42").Offset(0, date_col + 1).FormulaR1C1 = _
        "=IF(R[-30]C>0,R[-30]C,0) + IF(R[-27]C>0,R[-27]C,0) + IF(R[-24]C>0,R[-24]C,0) + " & Sheets(Portname).Range("A42").Offset(0, date_col + 1).Value
     
    
'Hide Old
Sheets(Portname).Range("A3:A55").Offset(0, date_col - 1).Value = Sheets(Portname).Range("A3:A55").Offset(0, date_col - 1).Value
Sheets(Portname).Columns("A:A").Offset(0, date_col - 3).EntireColumn.Hidden = True

'SCB
Sheets(Portname).Range("A5").Offset(0, date_col + 14).FormulaR1C1 = "=SCBAM_NextBusinessDay(R5C[-1])"
Sheets(Portname).Range("A11").Offset(0, date_col + 14).FormulaR1C1 = _
        "=IF(R5C>TODAY(),R[2]C[-1],SCBAM_GetSecDtl(""SCBSFF"",""-SACON SFF"",R5C[-1],""MarketVal""))"
Sheets(Portname).Range("A13").Offset(0, date_col + 14).FormulaR1C1 = "=R[-2]C+R[-1]C"
'Bay

Sheets(Portname).Range("A14").Offset(0, date_col + 14).FormulaR1C1 = _
        "=IF(R5C>TODAY(),RC[-1],SCBAM_GetSecDtl(""SCBSFF"",""CASH"",R5C[-1],""MarketVal""))"
Sheets(Portname).Range("A16").Offset(0, date_col + 14).FormulaR1C1 = "=R[-1]C+R[32]C-R[-3]C-R[2]C[-1]"
'UOBT
Sheets(Portname).Range("A17").Offset(0, date_col + 14).FormulaR1C1 = _
        "=IF(R5C>TODAY(),R[2]C[-1],SCBAM_GetSecDtl(""SCBSFF"",""INVSUOBTsff"",R5C[-1],""MarketVal""))"
Sheets(Portname).Range("A19").Offset(0, date_col + 14).FormulaR1C1 = "=R[-2]C+R[-1]C"

Sheets(Portname).Range("A21").Offset(0, date_col + 14).FormulaR1C1 = "=IF(R5C<=TODAY(),R[29]C[-1],R[27]C[-1])"
Sheets(Portname).Range("A35").Offset(0, date_col + 14).FormulaR1C1 = "=SUM(R[-11]C:R[-1]C)"
Sheets(Portname).Range("A47").Offset(0, date_col + 14).FormulaR1C1 = "=SUM(R[-9]C:R[-1]C)"
Sheets(Portname).Range("A48").Offset(0, date_col + 14).FormulaR1C1 = "=R[-27]C+R[-13]C-R[-1]C"
Sheets(Portname).Range("A51").Offset(0, date_col + 14).FormulaR1C1 = "=R[-1]C-R[-3]C"
Sheets(Portname).Range("A58").Offset(0, date_col + 14).FormulaR1C1 = _
        "=SUM(R[-34]C[1]:R[-31]C[1])-SUM(R[-20]C[1]:R[-19]C[1])"

Sheets(Portname).Range("A60").Offset(0, date_col + 14).FormulaR1C1 = _
        "=SCBAM_GetGroup1Dtl(""SCBSFF"",""NET ASSET VALUE"",R5C,""marketval"")+SCBAM_GetGroup1Dtl(""SCBSFF"",""NET ASSET VALUE"",R5C,""ai"")"
Sheets(Portname).Range("A61").Offset(0, date_col + 14).FormulaR1C1 = _
        "=IF(R5C>TODAY(),R61C[-1],R60C[-1])+SUM(R[-37]C:R[-34]C)-SUM(R[-23]C:R[-22]C)"

Sheets(Portname).Range("A62").Offset(0, date_col + 14).FormulaR1C1 = "=R48C/IF(R5C<TODAY(),R60C,R61C)"
Sheets(Portname).Activate
Sheets(Portname).Range("A5:A55").Offset(0, date_col + 13).Copy
Sheets(Portname).Range("A5:A55").Offset(0, date_col + 14).Select
    Selection.PasteSpecial Paste:=xlPasteFormats, Operation:=xlNone, _
        SkipBlanks:=False, Transpose:=False
Sheets(Portname).Range("A24").Offset(0, date_col + 2).Select


Application.Calculation = xlCalculationAutomatic


End Sub


