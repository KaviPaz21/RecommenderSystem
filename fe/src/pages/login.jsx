import { Hidden } from '@mui/material'
import React, { useState } from 'react'

export default function Login({activepagechanger , state}) {
    const [active , setactive] = React.useState(1)

    const [getuid , setgetuid] = useState(null)
    //relative left-1/2 top-1/2 border border-zinc-800  py-3 -translate-x-1/2 -translate-y-1/2 rounded-2xl
  return (
    <div className='w-[500px] '>
      <div className='text-center text-3xl font-semibold tracking-[10px] py-6 bg-gradient-to-b from-black via-rose-900 to-black'>SEPHORA</div>
 
      <div className='flex justify-center gap-12 mt-6'>
        <div className={`${active == 1? "border-b border-red-500":""} py-1 px-6 cursor-pointer `} onClick={() => setactive(1)}>Sign In</div>
        <div className={`${active == 2? "border-b border-red-500":""} py-1 px-6 cursor-pointer `} onClick={() => setactive(2)}>Join Now</div>
      </div>

      <div className='relative w-full h-[400px]  pt-12'>
        <div className={`${active ==1 ? "block" : 'hidden'} absolute px-12`}>
            <div className='text-zinc-400 text-sm pb-6'>Welcome back! Please enter your User ID to access your personalized beauty profile</div>

            <div>
                <div className='text-[14px] tracking-widest'>USER ID</div>
                <input type="text"  className='bg-black border mt-2 px-4 py-2 rounded-xl w-full border-stone-700' placeholder='User ID' onChange={(e)=>setgetuid(e.target.value)}/>
            </div>
            <div>
                <div className='text-center py-3 font-semibold bg-rose-700 rounded-xl mt-12 cursor-pointer hover:shadow-sm hover:shadow-rose-600 duration-300' onClick={()=>activepagechanger(2, getuid)}>{state}</div>
            </div>


            <div className='mt-12 text-stone-700 text-center'>
                <i>type : 23866342710 , 7542451569 ,2279900072 for existing users</i>
            </div>
        </div>
      </div>
    </div>
  )
}
