import logo from './logo.svg';
import './App.css';
import Login from './pages/login';
import Homepage from './pages/homepage';
import ProductPage from './pages/productPage';
import { useState } from 'react';
import axios from 'axios';

function App() {

  const [activepage , setactivepage ] = useState(1)
  const [userid , setuserid] = useState(null)
  const [profiledata , setprofiledata] = useState(null)
  const [state,  setstate] = useState("Access Account")
  const [userchoice , setuserchoice] = useState(null)
  const [toppicks , settoppicks] = useState(null)
  const [selectedprod , setselectedprod] = useState(null)
  const [similaritems, setsimilaritems] = useState(null)
  const be = "http://127.0.0.1:8000"

  const getuserdetails = async(userid)=>{
    setstate("Loading User data")
    const url = be + "/user-profile"
    const res = await axios.post(url , {"userid":userid})

    const results = await res.data
    //console.log(results)
    setprofiledata(results)
    
  }

  const gettingUserchoice = async(userid)=>{
    setstate("Getting User choices")
    const url = be + "/Recommendations/userchoice"
    const res = await axios.post(url , {"userid":userid})

    const results = await res.data
    if (results.message === "ok"){
        console.log(results)
        setuserchoice(results.items)
    }
    
  }
  const gettoppicks = async(userid)=>{
    setstate("Getting User choices")
    const url = be + "/Recommendations/toppicks"
    const res = await axios.post(url , {"userid":userid})

    const results = await res.data
    if (results.message === "ok"){
      console.log("top picks")
        //console.log(results)
        settoppicks(results.items)
    }
    
  }


  const getsimilarproducts = async(prodid)=>{
    setstate("Getting User choices")
    const url = be + "/Recommendations/similaritems"
    const res = await axios.post(url , {"userid":prodid})

    const results = await res.data
    if (results.message === "ok"){
      console.log("top picks")
        //console.log(results)
        setsimilaritems(results.items)
    }
    
  }
  




  const activepagechanger = (k, useridfromlogin)=>{
    if (k==2){
      setstate("Submitting...")
      setuserid(useridfromlogin)
      getuserdetails(useridfromlogin)
      gettingUserchoice(useridfromlogin)
      gettoppicks(useridfromlogin)
      setactivepage(k)
    }
    else{
    setactivepage(k)
    }
  }

  const viewproducts = (k, index , row,prodid)=>{
    if (row ==1){
      setselectedprod(userchoice[index])

    }else if(row ==2){
      setselectedprod(toppicks[index])
    }
    else if(row ==3){
      setselectedprod(similaritems[index])
    }
    getsimilarproducts(prodid)
    setactivepage(k)
  }
  return (
    <div className="bg-black w-screen h-screen text-white relative">
      <div className={`${activepage !=1 && "scale-0"} absolute left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2`}>
        <Login activepagechanger={activepagechanger} state={state}/>
      </div>

      <div className={`${activepage !=2 && "scale-0"} absolute `}>
        <Homepage 
          activepagechanger={activepagechanger} 
          profiledata={profiledata} 
          userchoice = {userchoice}
          userid={userid} 
          viewproducts={viewproducts}
          toppicks={toppicks}/>
          
      </div>
      <div className={`${activepage !=3 && "scale-0"} absolute `}>
        <ProductPage 
        similaritems={similaritems}
        selectedprod={selectedprod}
        viewproducts={viewproducts}
        activepagechanger={activepagechanger} />
      </div>
    </div>
  );
}

export default App;
